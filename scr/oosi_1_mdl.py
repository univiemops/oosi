# -*- coding: utf-8 -*-
"""Out-of-sample inference 1 - Modeling
v007

This Python script is an automated Machine Learning pipeline designed for
robust out-of-sample inference. It dynamically loads data and configurations to train,
tune, evaluate, and explain predictive models across different data types (continuous,
binary, or multiclass).

@author: david.steyrl@univie.ac.at
"""

import importlib.metadata as im
import logging
import math
import os
import pickle as pkl
import platform
import time
import warnings
from typing import Any, Dict, List, Tuple

import multiprocessing as mp
import numpy as np
import pandas as pd
import shapiq
import torch
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.utils import resample, shuffle
from sklearn.utils.validation import check_is_fitted
from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._sklearn.sklearn_utils import validate_data
from tabicl.shap import get_shapiq_explainer

from custom_splitter import RepeatedGroupKFold

# Suppress warnings that clog stdout/stderr during rapid solver iterations
WARNING_MSGS = [
    "y_pred contains classes not in y_true",
    "The max_iter was reached",
    "X does not have valid feature",
]
for msg in WARNING_MSGS:
    os.environ["PYTHONWARNINGS"] = f"ignore:{msg}:::"
    warnings.filterwarnings("ignore", message=msg)


# Monkey patching TabICLClassifier for predict_logits
def predict_logits(self, X: np.ndarray) -> np.ndarray:
    """Predict aggregated class logits for test samples.

    Applies the ensemble of TabICL models to make predictions, with each ensemble
    member providing predictions that are then averaged. The method:

    1. Transforms input data using the fitted encoders
    2. Applies the ensemble generator to create multiple views
    3. Forwards each view through the model
    4. Corrects for class shuffles
    5. Averages predictions across ensemble members

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples for prediction.  Columns that are entirely NaN are
        treated as masked features and excluded from inference.  This is
        useful for computing SHAP values, where masked features are
        represented as all-NaN columns.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
        Class logits for each test sample.
    """
    check_is_fitted(self)
    if isinstance(X, np.ndarray) and len(X.shape) == 1:
        # Reject 1D arrays to maintain sklearn compatibility
        raise ValueError("The provided input X is one-dimensional. Reshape your data.")

    # Check if prediction is possible
    has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
    has_training_data = (
        hasattr(self, "ensemble_generator_")
        and getattr(self.ensemble_generator_, "X_", None) is not None
    )
    if not has_kv_cache and not has_training_data:
        raise RuntimeError(
            "Cannot predict: this estimator was saved without training data and has no KV cache. "
            "Predictions require either cached KV projections or the original training data. "
            "Re-fit the estimator or load from a file saved with save_training_data=True or "
            "save_kv_cache=True."
        )

    if self.n_jobs is not None:
        assert self.n_jobs != 0
        old_n_threads = torch.get_num_threads()
        n_logical_cores = mp.cpu_count()

        if self.n_jobs > 0:
            if self.n_jobs > n_logical_cores:
                warnings.warn(
                    f"TabICL got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                    f" Only {n_logical_cores} threads will be used."
                )
            n_threads = max(n_logical_cores, self.n_jobs)
        else:
            n_threads = max(1, mp.cpu_count() + 1 + self.n_jobs)

        torch.set_num_threads(n_threads)

    # Preserve DataFrame structure to retain column names and types for correct feature transformation
    X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)

    # Detect all-NaN columns (used by SHAP's feature masking approach)
    if hasattr(X, "columns"):  # check for dataframe without importing pandas
        feature_mask = X.isna().all(axis=0).to_numpy()
    else:
        arr = np.asarray(X)
        if np.issubdtype(arr.dtype, np.number):
            feature_mask = np.isnan(arr).all(axis=0)
        else:
            # object dtype: v != v is True only for NaN in IEEE 754, safe for strings too
            feature_mask = np.array(
                [all(v != v for v in arr[:, i]) for i in range(arr.shape[1])]
            )

    if feature_mask is not None and not np.any(feature_mask):
        feature_mask = None

    # Fill masked columns so that transformers don't choke on NaN
    if feature_mask is not None:
        if hasattr(X, "columns"):  # Proxy way to check whether X is a dataframe
            X.iloc[:, feature_mask] = 0.0
        else:
            X[:, feature_mask] = 0.0

    X = self.X_encoder_.transform(X)

    # Skip KV cache when features are masked
    has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
    use_cache = has_kv_cache and feature_mask is None

    if use_cache:
        # Cache exists: forward only test data and use the pre-computed cache for training data
        test_data = self.ensemble_generator_.transform(X, mode="test")
        outputs = []
        for norm_method, (Xs_test,) in test_data.items():
            kv_cache = self.model_kv_cache_[norm_method]
            outputs.append(self._batch_forward_with_cache(Xs_test, kv_cache))
        outputs = np.concatenate(outputs, axis=0)
    else:
        # No cache or masked features: forward both training and test data
        data = self.ensemble_generator_.transform(
            X, mode="both", feature_mask=feature_mask
        )
        outputs = []
        for norm_method, (Xs, ys) in data.items():
            if feature_mask is None:
                feature_shuffles = self.ensemble_generator_.feature_shuffles_[
                    norm_method
                ]
            else:
                feature_shuffles = self.ensemble_generator_.masked_feature_shuffles_[
                    norm_method
                ]

            outputs.append(self._batch_forward(Xs, ys, feature_shuffles))
        outputs = np.concatenate(outputs, axis=0)

    # Extract class shuffle patterns from ensemble generator
    class_shuffles = []
    for shuffles in self.ensemble_generator_.class_shuffles_.values():
        class_shuffles.extend(shuffles)

    # Determine actual number of ensemble members
    n_estimators = len(class_shuffles)

    # Aggregate predictions from all ensemble members, correcting for class shuffles
    avg = np.zeros_like(outputs[0])
    for i, loop_shuffle in enumerate(class_shuffles):
        out = outputs[i]
        avg += out[..., loop_shuffle]

    # Calculate ensemble average
    avg /= n_estimators

    if self.n_jobs is not None:
        torch.set_num_threads(old_n_threads)

    return avg


TabICLClassifier.predict_logits = predict_logits


def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities for test samples.

    Uses predict_logits to get class logits and returns the class probabilities
    for each sample.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples for prediction.  Columns that are entirely NaN are
        treated as masked features and excluded from inference.  This is
        useful for computing SHAP values, where masked features are
        represented as all-NaN columns.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
        Class probabilities for each test sample.
    """
    avg = self.predict_logits(X)

    # Convert logits to probabilities
    if self.average_logits:
        avg = self.softmax(avg, axis=-1, temperature=self.softmax_temperature)

    # Normalize probabilities
    return avg / avg.sum(axis=1, keepdims=True)


TabICLClassifier.predict_proba = predict_proba


class LGBMLogOddsClassifier(LGBMClassifier):
    """
    A drop-in LGBMClassifier that outputs raw log-odds in predict_proba,
    fully compatible with scikit-learn search CV and shapiq.
    """

    def predict_proba(self, X, *args, **kwargs):
        # Clear out any internal 'raw_score' flags passed by predict()
        kwargs.pop("raw_score", None)
        # Get raw margin scores using the base model's predict method
        return super().predict_proba(X, raw_score=True, *args, **kwargs)


def setup_logging() -> None:
    """Configure simultaneous logging to both a project file and the terminal."""
    # Basic logging
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

    # Add console logger
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("Logging established")
    logging.info(f"Logs save at: {log_file}")


def document_requirements() -> None:
    """Document environmental dependencies, logging system configuration and Python specs."""
    lines = [f"{dist.name}=={dist.version}" for dist in im.distributions()]
    req_file = "../logs/requirements_oosi_1_mdl.txt"

    with open(req_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    logging.info(f"OS: {platform.system()} {platform.release()}")
    logging.info(f"Python Version: {platform.python_version()}")
    logging.info(f"Dependencies save at: {req_file}")


def get_configs_names(file_path: str) -> List[str]:
    """Load configurations names list from a specified YAML file."""
    with open(file_path, "r", encoding="utf-8") as file:
        configs_names = yaml.safe_load(file)["CONFIGS_NAMES"]

    logging.info(f"Config names loaded: {', '.join(configs_names)}")
    return configs_names


def get_configs_file_names(configs_names: List[str]) -> List[str]:
    """Convert configuration names into their respective YAML file paths."""
    config_file_names = [
        f"../configs/oosi_configs_{name}.yaml" for name in configs_names
    ]
    for name in config_file_names:
        logging.info(f"Config file name to load: {name}")
    return config_file_names


def get_configs(configs_file_names: List[str]) -> List[Dict[str, Any]]:
    """Load configuration details and load the associated dataset from files."""
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
        logging.info(f"Config and data loaded: {file_name}")

    return configs


def get_data(
    config: Dict[str, Any], y_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrieve features, target, and groups aligned on non-null targets."""
    y_df = config["Y"][y_name].to_frame().dropna()
    g_df = config["G"].reindex(index=y_df.index)
    x_df = config["X"].reindex(index=y_df.index)

    logging.info(f"Data sliced for {y_name}.")
    dropped = config["Y"].shape[0] - y_df.shape[0]
    if dropped > 0:
        logging.info(f"{dropped} samples dropped due to NaNs in {y_name}.")

    return x_df, y_df, g_df


def get_base_preprocessor(
    config: Dict[str, Any], target_type: str, scale_and_impute: bool = True
) -> ColumnTransformer:
    """Helper to assemble the common ColumnTransformer steps."""
    if scale_and_impute:
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
        low_cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                ("scaler", StandardScaler()),
            ]
        )
        high_cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target", TargetEncoder(target_type=target_type)),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        cont = Pipeline([("pass", "passthrough")])
        binary = Pipeline([("pass", "passthrough")])
        low_cat = Pipeline([("pass", "passthrough")])
        high_cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target", TargetEncoder(target_type=target_type)),
            ]
        )

    return ColumnTransformer(
        [
            ("cont", cont, config["X_CONTINUOUS_IND"]),
            ("bin", binary, config["X_BINARY_IND"]),
            ("low_cat", low_cat, config["X_LOW_CATEGORICAL_IND"]),
            ("high_cat", high_cat, config["X_HIGH_CATEGORICAL_IND"]),
        ]
    )


def get_linear_estimator_pipeline(
    config: Dict[str, Any], target_type: str, n_classes: int
) -> Tuple[Pipeline, Dict[str, Any], str]:
    """Construct a pipeline containing multi-type preprocessors and a linear estimator."""
    preprocessor = get_base_preprocessor(config, target_type, scale_and_impute=True)

    if target_type == "continuous":
        regressor = ElasticNet(max_iter=10000, warm_start=True, selection="random")
        predictor = TransformedTargetRegressor(
            regressor=regressor, transformer=StandardScaler()
        )
        space = {
            "predictor__regressor__alpha": loguniform(0.01, 10),
            "predictor__regressor__l1_ratio": uniform(0, 1),
        }
        scorer = "r2"
    elif target_type in ("binary", "multiclass"):
        predictor = LogisticRegression(
            class_weight="balanced", solver="saga", max_iter=10000, warm_start=True
        )
        space = {
            "predictor__C": loguniform(0.01, 10),
            "predictor__l1_ratio": uniform(0, 1),
        }
        scorer = "balanced_accuracy"
    else:
        raise ValueError(f"Unknown target_type: '{target_type}' encountered.")

    return (
        Pipeline([("preprocessor", preprocessor), ("predictor", predictor)]),
        space,
        scorer,
    )


def get_gb_estimator_pipeline(
    config: Dict[str, Any], target_type: str, n_classes: int
) -> Tuple[Pipeline, Dict[str, Any], str]:
    """Construct a pipeline containing preprocessors and a Gradient Boosting estimator."""
    preprocessor = get_base_preprocessor(config, target_type, scale_and_impute=False)
    lgbm_params = {
        "boosting_type": "gbdt",
        "num_leaves": 100,
        "max_depth": -1,
        "learning_rate": 0.01,
        "n_estimators": 1000,
        "subsample_for_bin": 10000,
        "min_split_gain": 0.0,
        "min_child_weight": 0.00001,
        "min_child_samples": 1,
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
        "verbosity": -1,
    }

    if target_type == "continuous":
        regressor = LGBMRegressor(objective="huber", **lgbm_params, **extra_params)
        predictor = TransformedTargetRegressor(
            regressor=regressor, transformer=StandardScaler()
        )
        space = {
            "predictor__regressor__colsample_bytree": uniform(0, 1),
            "predictor__regressor__extra_trees": [True, False],
            "predictor__regressor__path_smooth": loguniform(1, 100),
        }
        scorer = "r2"
    elif target_type in ("binary", "multiclass"):
        predictor = LGBMLogOddsClassifier(
            objective="multiclass",
            class_weight="balanced",
            **lgbm_params,
            **{**extra_params, "num_class": n_classes},
        )
        space = {
            "predictor__colsample_bytree": uniform(0, 1),
            "predictor__extra_trees": [True, False],
            "predictor__path_smooth": loguniform(1, 100),
        }
        scorer = "balanced_accuracy"
    else:
        raise ValueError(f"Unknown target_type: '{target_type}' encountered.")

    return (
        Pipeline([("preprocessor", preprocessor), ("predictor", predictor)]),
        space,
        scorer,
    )


def get_tabicl_estimator_pipeline(
    config: Dict[str, Any], target_type: str, n_classes: int
) -> Tuple[Pipeline, Dict[str, Any], str]:
    """Construct a pipeline containing preprocessors and a TabICL estimator."""
    if target_type == "continuous":
        return (
            TabICLRegressor(
                kv_cache=True,
                model_path="../models/tabicl-regressor-v2-20260212.ckpt",
                random_state=None,
            ),
            {},
            "r2",
        )
    elif target_type in ("binary", "multiclass"):
        return (
            TabICLClassifier(
                kv_cache=True,
                model_path="../models/tabicl-classifier-v2-20260212.ckpt",
                random_state=None,
            ),
            {},
            "balanced_accuracy",
        )
    raise ValueError(f"Unknown target_type: '{target_type}' encountered.")


def get_estimator_scores(
    estimator: Pipeline,
    x_tst: np.ndarray,
    y_tst: np.ndarray,
    target_type: str,
    sample_indices: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate performance of the estimator pipeline on holdout test data."""
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


def explain_pipeline(
    estimator: Pipeline,
    x_trn: np.ndarray,
    x_tst: np.ndarray,
    pipe_name: str,
    target_type: str,
    classes: List[Any] = None,
    budget: int = None,
    n_jobs: int = 1,
) -> Dict[Any, Any]:
    """Calculate feature contribution explanations using Permutation SHAP."""
    if pipe_name == "linear" and target_type in ["binary", "multiclass"]:
        pred_fun = estimator.decision_function
    elif pipe_name == "gb" and target_type in ["binary", "multiclass"]:
        pred_fun = estimator.predict_proba
    else:
        pred_fun = estimator
    explanations = {}
    for c_class in classes:
        if pipe_name in ["linear", "gb"]:
            explainer = shapiq.explainer.TabularExplainer(
                model=pred_fun,
                data=x_trn,
                class_index=c_class,
                imputer="marginal",
                approximator="auto",
                index="k-SII",
                max_order=2,
                random_state=None,
                verbose=False,
            )
        elif pipe_name == "tabicl":
            explainer = get_shapiq_explainer(
                estimator=pred_fun,
                data=x_trn,
                class_index=c_class,
                imputer="nan",
                index="k-SII",
                max_order=2,
                random_state=None,
            )
        else:
            raise ValueError(f"Pipeline name {pipe_name} was not in linear/gb/tabicl.")

        explanations[c_class] = explainer.explain_X(x_tst, n_jobs=n_jobs, budget=budget)
    return explanations


def log_metrics(scores_list: List[Dict[str, Any]], prefix: str = "") -> None:
    """Helper layout for clean and centralized metrics evaluation output logs."""
    label = f"Avg {prefix}".strip()
    if scores_list[0]["target_type"] == "continuous":
        logging.info(
            f"{label} Mean Absolute Error (MAE): {np.mean([i['mae'] for i in scores_list]):.2f}"
        )
        logging.info(
            f"{label} Coefficient of Determination (R²): {np.mean([i['r2'] for i in scores_list]) * 100:.2f}"
        )
    else:
        logging.info(
            f"{label} Balanced Accuracy: {np.mean([i['acc'] for i in scores_list]) * 100:.2f}"
        )


def downsample_if_needed(arr: np.ndarray, target_size: int, label: str) -> np.ndarray:
    """Drops rows containing NaNs, then applies scikit-learn random structural data subsetting allocations."""
    # Downsample if the remaining clean rows still exceed target_size
    if arr.shape[0] > target_size:
        logging.info(f"{label} downsampled to {target_size} samples for shapiq")
        return resample(arr, replace=False, n_samples=target_size)

    return arr


def main() -> None:
    """Coordinate configuration loads, orchestrate loops, and save results."""
    setup_logging()
    document_requirements()
    configs_names = get_configs_names("../configs/oosi_configs_names.yaml")
    configs = get_configs(get_configs_file_names(configs_names))
    pipeline_mapping = {
        "linear": get_linear_estimator_pipeline,
        "gb": get_gb_estimator_pipeline,
        "tabicl": get_tabicl_estimator_pipeline,
    }

    for config in configs:
        for target_idx, y_name in enumerate(config["Y_NAMES"]):
            analysis_name = f"{config['ANALYSIS_NAME']}__{y_name}"
            logging.info(f"Out-of-Sample-Inference started: {analysis_name}")
            x_df, y_df, g_df = get_data(config, y_name)

            # Map target metadata parameters seamlessly
            if target_idx in config["Y_CONTINUOUS_IND"]:
                target_type, classes, n_classes = "continuous", [None], None
            else:
                target_type = (
                    "binary" if target_idx in config["Y_BINARY_IND"] else "multiclass"
                )
                classes = y_df.squeeze().sort_values().unique().astype(int)
                n_classes = y_df.nunique()[y_name]

            logging.info(
                f"Target type: {target_type} | Classes: {classes} | Total Classes: {n_classes}"
            )
            n_features = len(config["X_NAMES"])

            for pipe_name in config["PIPELINE_CONFIGS"]:
                if pipe_name not in pipeline_mapping:
                    raise ValueError(
                        f"Pipeline name {pipe_name} was not in linear/gb/tabicl."
                    )

                logging.info(f"Out-of-Sample-Inference estimator: {pipe_name}")
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

                raw_estimator, search_space, scorer = pipeline_mapping[pipe_name](
                    config, target_type, n_classes
                )

                # Results infrastructure tracking metrics
                tracking = {
                    "": ([], []),
                    "_sh": ([], []),
                }  # Tuple footprint layout mapping (Scores, Shap)
                total_best_params = []
                total_x_tst_sample_shapiq = []

                cv = RepeatedGroupKFold(n_splits=5, n_repeats=20, random_state=3141592)

                for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(x_df, y=y_df, groups=g_df)
                ):
                    logging.info(
                        f"Fold {fold_idx + 1}: {config['ANALYSIS_NAME']} {y_name} {pipe_name}..."
                    )
                    start_time = time.time()

                    g_trn = g_df.iloc[train_idx].to_numpy().squeeze()
                    x_trn = x_df.iloc[train_idx].to_numpy().squeeze()
                    y_trn = y_df.iloc[train_idx].to_numpy().squeeze()
                    x_tst = x_df.iloc[test_idx].to_numpy().squeeze()
                    y_tst = y_df.iloc[test_idx].to_numpy().squeeze()

                    x_trn_sample = downsample_if_needed(
                        arr=x_trn, target_size=100, label="x_trn"
                    )
                    x_tst_sample = downsample_if_needed(
                        arr=x_tst, target_size=100, label="x_tst"
                    )

                    total_x_tst_sample_shapiq.append(x_tst_sample)

                    # Primary Hyperparameter Search Fit Phase
                    if pipe_name in ["linear", "gb"]:
                        search = RandomizedSearchCV(
                            estimator=raw_estimator,
                            param_distributions=search_space,
                            n_iter=100,
                            scoring=scorer,
                            n_jobs=-2,
                            cv=RepeatedGroupKFold(n_splits=5, n_repeats=20),
                        ).fit(X=x_trn, y=y_trn, groups=g_trn)
                        total_best_params.append(search.best_params_)
                        for k, v in search.best_params_.items():
                            logging.info(f"{k}: {v:.2f}")
                        estimator = search.best_estimator_
                    else:
                        estimator = raw_estimator.fit(X=x_trn, y=y_trn)

                    # Control Blueprint Shuffled Target setup paired inside structural loop
                    models_to_evaluate = {
                        "": (estimator, ""),
                        "_sh": (
                            clone(estimator).fit(X=x_trn, y=shuffle(y_trn)),
                            "Shuffle target ",
                        ),
                    }

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
                            budget=min(
                                2**n_features,
                                10 * n_features + math.comb(n_features, 2),
                            ),
                            n_jobs=1 if pipe_name == "tabicl" else -2,
                        )
                        tracking[suffix][1].append(shap_data)

                    # Save intermediate unified metrics arrays safely
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
                        f"Results saved. Fold run time: {time.time() - start_time:.2f}s"
                    )

    logging.info("Out-of-Sample-Inference finished.")


if __name__ == "__main__":
    main()
