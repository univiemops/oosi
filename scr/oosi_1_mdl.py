# -*- coding: utf-8 -*-
"""Out-of-sample inference 1 - Modeling
v010

@author: david.steyrl@univie.ac.at
"""

import gc
import importlib.metadata as im
import logging
import math
import os
import pickle as pkl
import platform
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, uniform
from shapiq.explainer import TabularExplainer
from shapiq.interaction_values import InteractionValues
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils import resample, shuffle
from tabicl import TabICLClassifier, TabICLRegressor
from tabicl.shap import get_shapiq_explainer

from custom_splitter import RepeatedGroupKFold, ExtendedGroupShuffleSplit

# Suppress warnings that clog stdout/stderr during rapid solver iterations
WARNING_MSGS = [
    "y_pred contains classes not in y_true",
    "The max_iter was reached",
    "X does not have valid feature names",
    "Not all budget is required",
    "The least populated class in y has only",
    "The sample size is larger than the number",
]

env_rules = [f"ignore:{msg}" for msg in WARNING_MSGS]
os.environ["PYTHONWARNINGS"] = ",".join(env_rules)

for msg in WARNING_MSGS:
    warnings.filterwarnings("ignore", message=f".*{msg}.*")


def setup_logging() -> None:
    """Set up the logging environment.

    Creates the target directory if it does not exist and configures both file
    and console handlers to log messages at the INFO level.

    Returns
    -------
    None
    """
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/oosi_1_mdl.log"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("[+] Logging established")
    logging.info(f" -> Logs save at: {log_file}")


def document_requirements() -> None:
    """Document system environment details and installed Python packages.

    Writes a list of installed packages and their versions to a text file
    and logs the current operating system and Python version.

    Returns
    -------
    None
    """
    lines = [f"{dist.name}=={dist.version}" for dist in im.distributions()]
    req_file = "../logs/requirements_oosi_1_mdl.txt"

    with open(req_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    logging.info("[+] Document requirements")
    logging.info(f" -> OS: {platform.system()} {platform.release()}")
    logging.info(f" -> Python Version: {platform.python_version()}")
    logging.info(f" -> Dependencies saved at: {req_file}")


def get_configs_names(file_path: str) -> list[str]:
    """Load configuration names from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML file containing the 'CONFIGS_NAMES' key.

    Returns
    -------
    configs_names : list of str
        List of configuration names retrieved from the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        configs_names = yaml.safe_load(file)["CONFIGS_NAMES"]

    logging.info(f"[+] Config names loaded: {', '.join(configs_names)}")
    return configs_names


def get_configs_file_names(configs_names: list[str]) -> list[str]:
    """Construct file paths for the configuration files based on config names.

    Parameters
    ----------
    configs_names : list of str
        List of configuration identifiers.

    Returns
    -------
    config_file_names : list of str
        List of constructed YAML configuration file paths.
    """
    config_file_names = [
        f"../configs/oosi_configs_{name}.yaml" for name in configs_names
    ]
    for name in config_file_names:
        logging.info(f" -> Config file name to load: {name}")
    return config_file_names


def get_configs(configs_file_names: list[str]) -> list[dict[str, Any]]:
    """Load configuration details and datasets specified in YAML files.

    Reads configuration parameters from each path, loads data from the
    specified Excel file and sheet, extracts groups (G), features (X),
    and targets (Y), and attaches them back to the configuration dictionary.

    Parameters
    ----------
    configs_file_names : list of str
        List of file paths to configuration files.

    Returns
    -------
    configs : list of dict of {str: typing.Any}
        List of loaded configuration dictionaries updated with DataFrame objects.
    """
    configs = []
    dtypes = {"G": np.int64, "X": np.float64, "Y": np.float64}

    for file_name in configs_file_names:
        with open(file_name, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        for key, col_key in [("G", "G_NAME"), ("X", "X_NAMES"), ("Y", "Y_NAMES")]:
            cols = config[col_key]
            df = pd.read_excel(
                config["PATH_TO_DATA"],
                sheet_name=config["SHEET_NAME"],
                usecols=cols,
                dtype=dtypes[key],
            )
            config[key] = df.reindex(cols, axis=1)

        configs.append(config)
        logging.info(f" -> Config and data loaded: {file_name}")

    return configs


def get_data(
    config: dict[str, Any], y_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract and align feature, target, and group DataFrames for a target variable.

    Drops missing values from the target variable and aligns feature (X) and
    group (G) DataFrames according to the target's non-NaN index.

    Parameters
    ----------
    config : dict of {str: typing.Any}
        Configuration dictionary containing 'X', 'Y', and 'G' DataFrames.
    y_name : str
        Name of the target column to extract.

    Returns
    -------
    x_df : pandas.DataFrame of shape (n_samples, n_features)
        Aligned feature matrix.
    y_df : pandas.DataFrame of shape (n_samples, 1)
        Target DataFrame without missing values.
    g_df : pandas.DataFrame of shape (n_samples, 1)
        Aligned group structure.
    """
    y_df = config["Y"][y_name].to_frame().dropna()
    g_df = config["G"].reindex(index=y_df.index)
    x_df = config["X"].reindex(index=y_df.index)

    logging.info(f" -> Data sliced for {y_name}.")
    dropped = config["Y"].shape[0] - y_df.shape[0]
    if dropped > 0:
        logging.info(f" --> {dropped} samples dropped due to NaNs in {y_name}.")

    return x_df, y_df, g_df


def get_preprocessor(
    config: dict[str, Any], target_type: str, pipe_name: str
) -> ColumnTransformer:
    """Create a ColumnTransformer preprocessor for continuous, binary, and categorical features.

    Parameters
    ----------
    config : dict of {str: typing.Any}
        Configuration dictionary containing column index lists ('X_CONTINUOUS_IND',
        'X_BINARY_IND', 'X_MULTICAT_IND').
    target_type : {'continuous', 'binary', 'multiclass'}
        The data type of the target variable, controlling target encoder behavior.
    pipe_name : {'linear', 'gb', 'tabicl'}
        Identifier for the modeling pipeline architecture.

    Returns
    -------
    preprocessor : sklearn.compose.ColumnTransformer
        Configured preprocessor applying target-specific transformations.
    """
    if pipe_name == "linear":
        cont = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        binary = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]
        )
        cat = Pipeline(
            [
                ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ("scaler", StandardScaler()),
            ]
        )
    elif pipe_name == "gb":
        cont = Pipeline([("scaler", StandardScaler())])
        binary = Pipeline([("scaler", StandardScaler())])
        cat = Pipeline(
            [
                (
                    "target",
                    TargetEncoder(
                        target_type=target_type,
                        cv=(
                            KFold(n_splits=5, shuffle=True, random_state=None)
                            if target_type == "continuous"
                            else StratifiedKFold(
                                n_splits=5, shuffle=True, random_state=None
                            )
                        ),
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
    elif pipe_name == "tabicl":
        cont = Pipeline([("passthrough", "passthrough")])
        binary = Pipeline([("passthrough", "passthrough")])
        cat = Pipeline(
            [
                (
                    "target",
                    TargetEncoder(
                        target_type=target_type,
                        cv=(
                            KFold(n_splits=5, shuffle=True, random_state=None)
                            if target_type == "continuous"
                            else StratifiedKFold(
                                n_splits=5, shuffle=True, random_state=None
                            )
                        ),
                    ),
                ),
            ]
        )

    return ColumnTransformer(
        [
            ("cont", cont, config["X_CONTINUOUS_IND"]),
            ("bin", binary, config["X_BINARY_IND"]),
            ("cat", cat, config["X_MULTICAT_IND"]),
        ]
    )


def get_estimator_pipeline(
    config: dict[str, Any], target_type: str, n_classes: int, pipe_name: str
) -> tuple[Pipeline, dict[str, Any], str]:
    """Construct estimator pipelines, parameter spaces, and evaluation metrics.

    Parameters
    ----------
    config : dict of {str: typing.Any}
        Configuration dictionary containing dataset metadata and parameters.
    target_type : {'continuous', 'binary', 'multiclass'}
        The target data type indicating regression or classification.
    n_classes : int or None
        Number of target classes (for classification models).
    pipe_name : {'linear', 'gb', 'tabicl'}
        Type of pipeline to construct.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Scikit-learn pipeline comprising preprocessor and estimator.
    space : dict of {str: typing.Any}
        Hyperparameter search space dictionary for model tuning.
    scorer : str
        Scoring metric name (e.g., 'r2' or 'balanced_accuracy').
    """
    preprocessor = get_preprocessor(config, target_type, pipe_name=pipe_name)
    scorer = "r2" if target_type == "continuous" else "balanced_accuracy"

    if pipe_name == "linear":
        if target_type == "continuous":
            regressor = ElasticNet(max_iter=10000, warm_start=True, selection="random")
            predictor = TransformedTargetRegressor(
                regressor=regressor, transformer=StandardScaler()
            )
            space = {
                "predictor__regressor__alpha": loguniform(0.01, 100),
                "predictor__regressor__l1_ratio": uniform(0, 1),
            }
        elif target_type in ("binary", "multiclass"):
            predictor = LogisticRegression(
                class_weight="balanced", solver="saga", max_iter=10000, warm_start=True
            )
            space = {
                "predictor__C": loguniform(0.01, 100),
                "predictor__l1_ratio": uniform(0, 1),
            }

    elif pipe_name == "gb":
        lgbm_params = {
            "boosting_type": "gbdt",
            "num_leaves": 100,
            "max_depth": -1,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "subsample_for_bin": 10000,
            "min_split_gain": 0.0,
            "min_child_weight": 0.00001,
            "min_child_samples": 2,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "importance_type": "gain",
        }

        extra_params = {
            "bagging_seed": None,
            "data_random_seed": None,
            "data_sample_strategy": "bagging",
            "extra_seed": None,
            "feature_fraction_seed": None,
            "feature_pre_filter": False,
            "force_col_wise": True,
            "min_data_in_bin": 1,
            "min_data_in_leaf": 2,
            "verbosity": -1,
        }

        if target_type == "continuous":
            regressor = LGBMRegressor(
                objective="mean_squared_error", **lgbm_params, **extra_params
            )
            predictor = TransformedTargetRegressor(
                regressor=regressor, transformer=StandardScaler()
            )
            space = {
                "predictor__regressor__colsample_bytree": uniform(0, 1),
                "predictor__regressor__extra_trees": [True, False],
                "predictor__regressor__path_smooth": loguniform(0.1, 100),
            }
        elif target_type in ("binary", "multiclass"):
            predictor = LGBMClassifier(
                objective="multiclass",
                class_weight="balanced",
                **lgbm_params,
                **{**extra_params, "num_class": n_classes},
            )
            space = {
                "predictor__colsample_bytree": uniform(0.1, 0.9),
                "predictor__extra_trees": [True, False],
                "predictor__path_smooth": loguniform(0.1, 100),
            }

    elif pipe_name == "tabicl":
        space = {}
        if target_type == "continuous":
            predictor = TabICLRegressor(
                n_estimators=10,
                batch_size=None,
                kv_cache=True,
                model_path="../models/tabicl-regressor-v2-20260212.ckpt",
                random_state=None,
            )
        elif target_type in ("binary", "multiclass"):
            predictor = TabICLClassifier(
                n_estimators=10,
                batch_size=None,
                kv_cache=True,
                model_path="../models/tabicl-classifier-v2-20260212.ckpt",
                random_state=None,
            )

    return (
        Pipeline([("preprocessor", preprocessor), ("predictor", predictor)]),
        space,
        scorer,
    )


def get_splitter(config: dict[str, Any]):
    """Instantiate CV splitter dynamically based on YAML configurations.

    Parameters
    ----------
    config : dict of {str: typing.Any}
        Configuration dictionary containing 'SPLITTER' and 'SPLITTER_PARAMS'.

    Returns
    -------
    splitter : object
        Cross-validation splitter instance (e.g., RepeatedGroupKFold or
        ExtendedGroupShuffleSplit).
    """
    splitter_name = config["SPLITTER"]
    params = config["SPLITTER_PARAMS"]

    if splitter_name == "RepeatedGroupKFold":
        return RepeatedGroupKFold(**params)
    elif splitter_name == "ExtendedGroupShuffleSplit":
        return ExtendedGroupShuffleSplit(**params)


def get_estimator_scores(
    estimator: Pipeline,
    x_tst: np.ndarray,
    y_tst: np.ndarray,
    target_type: str,
    sample_indices: np.ndarray,
) -> dict[str, Any]:
    """Evaluate fitted estimator performance on test dataset and compile metrics.

    Parameters
    ----------
    estimator : sklearn.pipeline.Pipeline
        Fitted pipeline estimator used to generate predictions.
    x_tst : numpy.ndarray of shape (n_samples, n_features)
        Test feature data matrix.
    y_tst : numpy.ndarray of shape (n_samples,)
        True target labels or continuous targets.
    target_type : {'continuous', 'binary', 'multiclass'}
        Target classification type used to compute relevant scoring metrics.
    sample_indices : numpy.ndarray of shape (n_samples,)
        Indices of test samples used within the split.

    Returns
    -------
    scores : dict of {str: typing.Any}
        Dictionary containing sample indices, actual targets, predictions,
        and performance metrics (MAE, R2, accuracy).
    """
    y_pred = estimator.predict(x_tst)

    return {
        "sample_ind": sample_indices,
        "y_tst": y_tst,
        "y_pred": y_pred,
        "mae": (
            mean_absolute_error(y_tst, y_pred) if target_type == "continuous" else None
        ),
        "r2": r2_score(y_tst, y_pred) if target_type == "continuous" else None,
        "acc": (
            balanced_accuracy_score(y_tst, y_pred)
            if not target_type == "continuous"
            else None
        ),
        "target_type": target_type,
    }


def group_features(
    iv: InteractionValues, features_to_group: list[int] or set[int], new_feature_id: int
) -> InteractionValues:
    """Group subset indices within a SHAP interaction values object into a unified index.

    Parameters
    ----------
    iv : shapiq.interaction_values.InteractionValues
        Original InteractionValues instance.
    features_to_group : list of int or set of int
        Indices of features to combine into a single index entity.
    new_feature_id : int
        New integer index assigned to the grouped feature set.

    Returns
    -------
    grouped_iv : shapiq.interaction_values.InteractionValues
        New InteractionValues object with aggregated interaction features.
    """
    group = set(features_to_group)
    new_values = {}

    for subset, value in iv.dict_values.items():
        subset_set = set(subset)
        intersection = subset_set.intersection(group)

        if len(intersection) == 0:
            new_values[subset] = new_values.get(subset, 0.0) + value
        else:
            outside_elements = subset_set.difference(group)
            new_subset = tuple(sorted(list(outside_elements) + [new_feature_id]))
            new_values[new_subset] = new_values.get(new_subset, 0.0) + value

    baseline_val = iv.baseline_value
    new_n_players = iv.n_players - len(group) + 1
    new_min_order = (
        min(len(k) for k in new_values.keys()) if new_values else iv.min_order
    )

    return InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=iv.max_order,
        min_order=new_min_order,
        n_players=new_n_players,
        baseline_value=baseline_val,
        estimation_budget=iv.estimation_budget,
    )


def reorder_interaction_values(
    iv: InteractionValues, indices: list[int]
) -> InteractionValues:
    """Remap target indices within an InteractionValues instance according to a sequence map.

    Parameters
    ----------
    iv : shapiq.interaction_values.InteractionValues
        Original interaction values structure to reorder.
    indices : list of int
        Mapping list where index positions define new player indices.

    Returns
    -------
    reordered_iv : shapiq.interaction_values.InteractionValues
        InteractionValues instance with remapped player feature indices.

    Raises
    ------
    ValueError
        If length of provided indices list is smaller than total number of players.
    IndexError
        If an index present inside interaction subset is out of bounds for target mapping.
    """
    new_values = {}
    n_players_current = iv.n_players

    if len(indices) < n_players_current:
        raise ValueError(
            f"The length of the indices list ({len(indices)}) must be at least "
            f"the number of players in the InteractionValues object ({n_players_current})."
        )

    for subset, value in iv.dict_values.items():
        try:
            new_subset = tuple(sorted(indices[p] for p in subset))
        except IndexError as e:
            raise IndexError(
                f"Player index in subset {subset} is out of bounds for the provided indices list."
            ) from e
        new_values[new_subset] = value

    baseline_val = getattr(iv, "baseline_value", 0.0)
    est_budget = getattr(iv, "estimation_budget", None)
    is_estimated = getattr(iv, "estimated", False)

    return InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=iv.max_order,
        min_order=iv.min_order,
        n_players=len(indices),
        baseline_value=baseline_val,
        estimation_budget=est_budget,
        estimated=is_estimated,
    )


def explain_pipeline(
    estimator: Pipeline,
    x_trn: np.ndarray,
    x_tst: np.ndarray,
    pipe_name: str,
    target_type: str,
    classes: list[Any] = None,
    cat_ind: list[Any] = None,
    x_names_sequence: list[Any] = None,
) -> dict[Any, Any]:
    """Compute feature interaction explanations using SHAPIQ for model evaluation pipelines.

    Parameters
    ----------
    estimator : sklearn.pipeline.Pipeline
        Fitted model pipeline to explain.
    x_trn : numpy.ndarray of shape (n_samples_train, n_features)
        Training dataset sample used for background baseline.
    x_tst : numpy.ndarray of shape (n_samples_test, n_features)
        Test dataset sample to explain.
    pipe_name : {'linear', 'gb', 'tabicl'}
        Identifier string of the model architecture.
    target_type : {'continuous', 'binary', 'multiclass'}
        Data type of the target variable.
    classes : list of Any, default=None
        List of target class indices to calculate explanations for.
    cat_ind : list of Any, default=None
        Indices of categorical features requiring re-grouping post-transformation.
    x_names_sequence : list of Any, default=None
        Full feature index sequence used for ordering interaction values.

    Returns
    -------
    explanations : dict of {Any: list of shapiq.interaction_values.InteractionValues}
        Mapping of target class indices to lists of InteractionValues for each instance.
    """
    if pipe_name == "linear" and target_type in ["binary", "multiclass"]:
        pred_fun = estimator.decision_function
    elif pipe_name == "gb" and target_type in ["binary", "multiclass"]:
        pred_fun = estimator.predict_proba
    else:
        pred_fun = estimator

    n_features = estimator[0].transform(x_tst).shape[1]

    explanations = {}
    for c_class in classes:
        if pipe_name in ["linear", "gb"]:
            explainer = TabularExplainer(
                model=pred_fun,
                data=x_trn,
                class_index=c_class,
                imputer="marginal",
                approximator="auto",
                index="k-SII",
                max_order=2,
                random_state=None,
                verbose=False,
                sample_size=None,
            )
            explanations[c_class] = explainer.explain_X(
                x_tst,
                n_jobs=-2,
                budget=min(2**n_features, 10*n_features + 10*math.comb(n_features, 2)),
            )
        elif pipe_name == "tabicl":
            explainer = get_shapiq_explainer(
                estimator=estimator[1],
                data=estimator[0].transform(x_trn),
                imputer="nan",
                index="k-SII",
                max_order=2,
                class_index=c_class,
            )
            explanations[c_class] = explainer.explain_X(
                estimator[0].transform(x_tst),
                n_jobs=1,
                budget=min(2**n_features, 10*n_features + 10*math.comb(n_features, 2)),
            )

    if target_type == "multiclass" and pipe_name == "tabicl":
        n_cat = len(cat_ind)
        n_features_orig = x_trn.shape[1]
        for class_key in explanations.keys():
            iv_list = []
            for iv in explanations[class_key]:
                for num, _ in enumerate(cat_ind):
                    start_ind = n_features - ((n_cat - num) * len(classes))
                    end_ind = start_ind + len(classes)
                    group_ind = list(range(start_ind, end_ind))
                    new_ind = n_features_orig - (n_cat - num)
                    iv = group_features(
                        iv=iv, features_to_group=group_ind, new_feature_id=new_ind
                    )
                iv_list.append(iv)
            explanations[class_key] = iv_list

    if pipe_name == "tabicl":
        for class_key in explanations.keys():
            iv_list = []
            for iv in explanations[class_key]:
                iv_list.append(reorder_interaction_values(iv, x_names_sequence))
            explanations[class_key] = iv_list

    return explanations


def log_metrics(scores_list: list[dict[str, Any]], prefix: str = "") -> None:
    """Calculate and log aggregate performance metrics from evaluation scores.

    Parameters
    ----------
    scores_list : list of dict of {str: typing.Any}
        List of target prediction score dictionaries generated by `get_estimator_scores`.
    prefix : str, default=""
        Prefix label string used in formatted log output messages.

    Returns
    -------
    None
    """
    label = f"Avg {prefix}".strip()

    if scores_list[0]["target_type"] == "continuous":
        logging.info(
            f" --> {label} Mean Absolute Error (MAE): {np.mean([i['mae'] for i in scores_list]):.2f}"
        )
        logging.info(
            f" --> {label} Coefficient of Determination (R²): {np.mean([i['r2'] for i in scores_list]) * 100:.2f}"
        )
    else:
        logging.info(
            f" --> {label} Balanced Accuracy: {np.mean([i['acc'] for i in scores_list]) * 100:.2f}"
        )


def downsample_if_needed(arr: np.ndarray, target_size: int, label: str) -> np.ndarray:
    """Downsample data array without replacement if its length exceeds target size.

    Parameters
    ----------
    arr : numpy.ndarray of shape (n_samples, ...)
        Input data array to downsample.
    target_size : int
        Maximum permitted number of rows/samples.
    label : str
        Description of input dataset for logging purposes.

    Returns
    -------
    downsampled_arr : numpy.ndarray of shape (n_samples_sampled, ...)
        Original data array or downsampled sample array.
    """
    if arr.shape[0] > target_size:
        logging.info(f" -> {label} downsampled to {target_size} samples for shapiq")
        return resample(arr, replace=False, n_samples=target_size, random_state=3141592)

    return arr


def main() -> None:
    """Execute the core out-of-sample inference model pipeline and logging routine.

    Returns
    -------
    None
    """
    setup_logging()
    document_requirements()
    configs_names = get_configs_names("../configs/oosi_configs_names.yaml")
    configs = get_configs(get_configs_file_names(configs_names))

    for config in configs:
        for target_idx, y_name in enumerate(config["Y_NAMES"]):
            analysis_name = f"{config['ANALYSIS_NAME']}__{y_name}"
            logging.info(f"[+] Out-of-Sample-Inference started: {analysis_name}")
            x_df, y_df, g_df = get_data(config, y_name)

            if target_idx in config["Y_CONTINUOUS_IND"]:
                target_type, classes, n_classes = "continuous", [None], None
            else:
                target_type = (
                    "binary" if target_idx in config["Y_BINARY_IND"] else "multiclass"
                )
                classes = y_df.squeeze().sort_values().unique().astype(int)
                n_classes = y_df.nunique()[y_name]

            logging.info(
                f" -> Target type: {target_type} | Classes: {classes} | Total Classes: {n_classes}"
            )

            for pipe_name in config["PIPELINE_CONFIGS"]:
                logging.info(f" -> Out-of-Sample-Inference estimator: {pipe_name}")
                save_path = f"../results/{analysis_name}/{pipe_name}"
                os.makedirs(save_path, exist_ok=True)

                with open(f"{save_path}/config.pkl", "wb") as f:
                    pkl.dump(config, f)
                with open(f"{save_path}/params.pkl", "wb") as f:
                    pkl.dump(
                        {
                            "y_name": y_name,
                            "target_type": target_type,
                            "classes": classes,
                            "n_classes": n_classes,
                            "pipe_name": pipe_name,
                        },
                        f,
                    )

                tracking = {
                    "": ([], []),
                    "_sh": ([], []),
                }
                total_best_params = []
                total_x_tst_sample_shapiq = []

                cv = get_splitter(config)

                for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(x_df, y=y_df, groups=g_df)
                ):
                    logging.info(
                        f"[+] Fold {fold_idx + 1}: {config['ANALYSIS_NAME']} {y_name} {pipe_name}..."
                    )
                    start_time = time.time()

                    g_trn = g_df.iloc[train_idx].to_numpy().squeeze()
                    x_trn = x_df.iloc[train_idx].to_numpy().squeeze()
                    y_trn = y_df.iloc[train_idx].to_numpy().squeeze()
                    g_tst = g_df.iloc[test_idx].to_numpy().squeeze()  # noqa
                    x_tst = x_df.iloc[test_idx].to_numpy().squeeze()
                    y_tst = y_df.iloc[test_idx].to_numpy().squeeze()

                    raw_estimator, search_space, scorer = get_estimator_pipeline(
                        config, target_type, n_classes, pipe_name
                    )

                    if pipe_name in ["linear", "gb"]:
                        search = RandomizedSearchCV(
                            estimator=raw_estimator,
                            param_distributions=search_space,
                            n_iter=100,
                            scoring=scorer,
                            n_jobs=-2,
                            cv=RepeatedGroupKFold(n_splits=5, n_repeats=6),
                        ).fit(X=x_trn, y=y_trn, groups=g_trn)
                        total_best_params.append(search.best_params_)
                        for k, v in search.best_params_.items():
                            logging.info(f" -> {k}: {v:.2f}")
                        estimator = search.best_estimator_
                    else:
                        estimator = raw_estimator.fit(X=x_trn, y=y_trn)

                    models_to_evaluate = {
                        "": (estimator, ""),
                        "_sh": (
                            clone(estimator).fit(X=x_trn, y=shuffle(y_trn)),
                            "Shuffle target ",
                        ),
                    }

                    x_trn_sample = downsample_if_needed(
                        arr=x_trn, target_size=1000, label="x_trn"
                    )
                    x_tst_sample = downsample_if_needed(
                        arr=x_tst, target_size=100, label="x_tst"
                    )
                    total_x_tst_sample_shapiq.append(x_tst_sample)

                    for suffix, (mdl, log_prefix) in models_to_evaluate.items():
                        score_metrics = get_estimator_scores(
                            mdl, x_tst, y_tst, target_type, test_idx
                        )
                        tracking[suffix][0].append(score_metrics)
                        log_metrics(tracking[suffix][0], prefix=log_prefix)

                        shap_data = explain_pipeline(
                            estimator=mdl,
                            x_trn=x_trn_sample,
                            x_tst=x_tst_sample,
                            pipe_name=pipe_name,
                            target_type=target_type,
                            classes=classes,
                            cat_ind=config["X_MULTICAT_IND"],
                            x_names_sequence=config["X_CONTINUOUS_IND"]
                            + config["X_BINARY_IND"]
                            + config["X_MULTICAT_IND"],
                        )
                        tracking[suffix][1].append(shap_data)

                    files_to_save = {
                        "total_best_params.pkl": total_best_params,
                        "total_scores.pkl": tracking[""][0],
                        "total_shapii.pkl": tracking[""][1],
                        "total_scores_sh.pkl": tracking["_sh"][0],
                        "total_shapii_sh.pkl": tracking["_sh"][1],
                        "total_x_tst_sample_shapiq.pkl": np.vstack(
                            total_x_tst_sample_shapiq
                        ),
                    }

                    for filename, datasets in files_to_save.items():
                        with open(f"{save_path}/{filename}", "wb") as f:
                            pkl.dump(datasets, f)

                    logging.info(
                        f"[✓] Results saved. Fold run time: {time.time() - start_time:.2f}s"
                    )

                    del raw_estimator
                    del estimator
                    del models_to_evaluate
                    del mdl
                    gc.collect()
                    torch.cuda.empty_cache()

    logging.info("[✓] Out-of-Sample-Inference finished.")


if __name__ == "__main__":
    main()
