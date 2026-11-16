# -*- coding: utf-8 -*-
"""Out-of-sample inference 3 - Exploratory data analysis
v003

@author: david.steyrl@univie.ac.at
"""

from __future__ import annotations

import importlib.metadata as metadata
import logging
import platform
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import gaussian_kde, norm
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLClassifier, TabICLRegressor, TabICLUnsupervised

from custom_splitter import RepeatedGroupKFold

# Project paths are relative to the project working directory.
LOG_DIR = Path("../logs")
CONFIG_NAMES_PATH = Path("../configs/oosi_configs_names.yaml")
CONFIG_DIR = Path("../configs")
DATA_OUTPUT_DIR = Path("../eda")

RANDOM_STATE = 3141592
N_CV_SPLITS = 5
N_CV_REPEATS = 2
N_ESTIMATORS = 10
N_OUTLIER_PERMUTATIONS = 100
DEFAULT_CONTAMINATION = 0.01
FIGURE_DPI = 300


def setup_logging() -> None:
    """Configure logging to write to both a project log and the terminal.

    Creates the log directory when it does not yet exist. Existing log files are
    overwritten at the beginning of each run.

    Returns
    -------
    None
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "oosi_3_eda.log"

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("Logging established.")
    logging.info("Log file: %s", log_path)


def document_requirements() -> None:
    """Save installed package versions and basic runtime information.

    Writes the installed distributions to ``../logs/requirements_oosi_3_eda.txt``.

    Returns
    -------
    None
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    requirements_path = LOG_DIR / "requirements_oosi_3_eda.txt"

    # Record the complete environment to make analysis runs reproducible.
    requirements = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    )
    requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    logging.info("Operating system: %s %s", platform.system(), platform.release())
    logging.info("Python version: %s", platform.python_version())
    logging.info("Package versions saved to: %s", requirements_path)


def get_config_names(file_path: str | Path) -> list[str]:
    """Load the configuration names from a YAML file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the YAML file containing the ``CONFIGS_NAMES`` key.

    Returns
    -------
    list of str
        Configuration names to load.
    """
    with Path(file_path).open("r", encoding="utf-8") as file:
        config_names = yaml.safe_load(file)["CONFIGS_NAMES"]

    logging.info("Configuration names: %s", ", ".join(config_names))
    return config_names


def get_config_file_paths(config_names: list[str]) -> list[Path]:
    """Construct the YAML paths corresponding to configuration names.

    Parameters
    ----------
    config_names : list of str
        Configuration names without the ``oosi_configs_`` prefix.

    Returns
    -------
    list of pathlib.Path
        Paths to the configuration files.
    """
    paths = [CONFIG_DIR / f"oosi_configs_{name}.yaml" for name in config_names]
    for path in paths:
        logging.info("Configuration file: %s", path)
    return paths


def get_configs(config_file_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load YAML configurations and the corresponding Excel data.

    Parameters
    ----------
    config_file_paths : list of str or pathlib.Path
        Paths to the project configuration files.

    Returns
    -------
    list of dict
        Configuration dictionaries augmented with ``G``, ``X``, and ``Y``
        DataFrames containing group, feature, and target data, respectively.
    """
    configs = []
    data_types = {"G": np.int64, "X": np.float64, "Y": np.float64}
    column_keys = {"G": "G_NAME", "X": "X_NAMES", "Y": "Y_NAMES"}

    for file_path in config_file_paths:
        with Path(file_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Load each group of columns separately to preserve the intended dtypes.
        for data_key, config_key in column_keys.items():
            columns = config[config_key]
            data = pd.read_excel(
                config["PATH_TO_DATA"],
                sheet_name=config["SHEET_NAME"],
                usecols=columns,
                dtype=data_types[data_key],
            )
            config[data_key] = data.reindex(columns=columns)

        configs.append(config)
        logging.info("Configuration and data loaded: %s", file_path)

    return configs


def get_data(
    config: dict[str, Any],
    y_name: str,
    drop_target_na: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return features, one target, and grouping data with aligned indices.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the ``X``, ``Y``, and ``G``
        DataFrames.
    y_name : str
        Name of the target column to retrieve.
    drop_target_na : bool, default=True
        Whether to remove observations with a missing target value.

    Returns
    -------
    tuple of pandas.DataFrame
        Feature, target, and group DataFrames with matching row indices.
    """
    if y_name not in config["Y"].columns:
        raise KeyError(f"Target column {y_name!r} is not present in config['Y'].")

    y_data = config["Y"][[y_name]].copy()
    if drop_target_na:
        y_data = y_data.dropna(subset=[y_name])

    x_data = config["X"].reindex(index=y_data.index)
    group_data = config["G"].reindex(index=y_data.index)

    if group_data.shape[1] != 1:
        raise ValueError("Exactly one grouping variable is required.")

    n_dropped = len(config["Y"]) - len(y_data)
    logging.info("-> Data sliced for target %s.", y_name)
    if n_dropped:
        logging.info(
            "-> %d observations removed due to missing target values.", n_dropped
        )

    return x_data, y_data, group_data


def _save_and_show_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save, display, and close a Matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save and display.
    output_path : pathlib.Path
        Destination PNG path.

    Returns
    -------
    None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _format_heatmap(
    matrix: np.ndarray | pd.DataFrame,
    labels: list[str] | pd.Index,
    title: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    """Draw and save a consistently formatted square heatmap.

    Parameters
    ----------
    matrix : array-like
        Values to display in the heatmap.
    labels : list of str or pandas.Index
        Labels for both matrix axes.
    title : str
        Figure title.
    colorbar_label : str
        Label displayed beside the colorbar.
    output_path : pathlib.Path
        Destination PNG path.

    Returns
    -------
    None
    """
    max_label_length = max(map(len, labels))
    n_features = len(labels)
    figure_size = (
        n_features * 0.6 + max_label_length * 0.1 + 1,
        n_features * 0.6 + max_label_length * 0.1 + 1,
    )
    fig, ax = plt.subplots(figsize=figure_size)

    sns.heatmap(
        matrix,
        vmin=0,
        vmax=1,
        cmap="Greys",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=True,
        cbar_kws={"label": colorbar_label, "shrink": 0.6},
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.axes[-1].tick_params(labelsize=10)
    fig.axes[-1].set_ylabel(colorbar_label, fontsize=10)
    _save_and_show_figure(fig, output_path)


def plot_1d_distribution(
    config: dict[str, Any], y_name: str, store_path: str | Path
) -> None:
    """Plot univariate distributions for all features and the selected target.

    Parameters
    ----------
    config : dict
        Analysis configuration and loaded data.
    y_name : str
        Target column used to select the analysis sample.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    logging.info("[+] Plotting univariate distributions.")
    x_data, y_data, _ = get_data(config, y_name)
    data = pd.concat([x_data, y_data], axis=1)

    max_name_length = max(map(len, data.columns))
    n_columns = len(data.columns)
    figure_size = (max_name_length * 0.1 + 5, (n_columns + 1) * 1.1 + 1)
    fig, axes = plt.subplots(
        nrows=n_columns,
        ncols=1,
        figsize=figure_size,
        sharex=False,
        squeeze=False,
    )

    # Use one horizontal violin plot per variable for comparable summaries.
    for ax, column in zip(axes[:, 0], data.columns):
        sns.violinplot(
            x=data[column],
            bw_method="scott",
            bw_adjust=0.25,
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=1,
            color="#777777",
            saturation=1.0,
            ax=ax,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(
            column,
            rotation="horizontal",
            horizontalalignment="right",
            fontsize=10,
        )
        ax.set_xlabel("")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)

    axes[-1, 0].set_xlabel("Range", fontsize=10)
    fig.suptitle(
        f"{config['ANALYSIS_NAME']}__{y_name}\n1D Data Distributions\n",
        fontsize=10,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    _save_and_show_figure(fig, Path(store_path) / "1_1D_distributions.png")

    logging.info("[✓] Plotting univariate distributions finished.")


def plot_2d_distribution(
    config: dict[str, Any], y_name: str, store_path: str | Path
) -> None:
    """Plot pairwise feature and target distributions.

    Parameters
    ----------
    config : dict
        Analysis configuration and loaded data.
    y_name : str
        Target column used to select the analysis sample.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    logging.info("[+] Plotting pairwise distributions.")
    x_data, y_data, _ = get_data(config, y_name)
    data = pd.concat([x_data, y_data], axis=1)

    pair_plot = sns.pairplot(
        data,
        corner=False,
        diag_kind="kde",
        plot_kws={"color": "#777777"},
        diag_kws={"color": "#777777"},
    )
    pair_plot.map_lower(sns.kdeplot, levels=3, color=".2")
    pair_plot.fig.suptitle(
        f"{config['ANALYSIS_NAME']}__{y_name}\n2D Data Distributions\n",
        fontsize=10,
        y=1.0,
    )
    _save_and_show_figure(pair_plot.fig, Path(store_path) / "2_2D_distributions.png")

    logging.info("[✓] Plotting pairwise distributions finished.")


def plot_bivariate_correlation(
    config: dict[str, Any], y_name: str, store_path: str | Path
) -> None:
    """Plot absolute pairwise correlations for features and the target.

    Parameters
    ----------
    config : dict
        Analysis configuration and loaded data.
    y_name : str
        Target column used to select the analysis sample.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    logging.info("[+] Plotting bivariate correlations.")
    x_data, y_data, _ = get_data(config, y_name)
    data = pd.concat([x_data, y_data], axis=1)
    correlations = data.corr(numeric_only=True).abs()

    _format_heatmap(
        matrix=correlations,
        labels=correlations.columns,
        title=(
            f"{config['ANALYSIS_NAME']}__{y_name}\n"
            "Bivariate Correlations (Absolute Values)\n"
        ),
        colorbar_label="|Correlation| (0 to 1)",
        output_path=Path(store_path) / "3_bivariate_correlations.png",
    )

    logging.info("[✓] Plotting bivariate correlations finished.")


def _get_column_objectives(
    config: dict[str, Any], x_columns: list[str], y_name: str
) -> dict[str, str]:
    """Map each feature and target column to its prediction objective.

    Parameters
    ----------
    config : dict
        Analysis configuration containing feature and target type indices.
    x_columns : list of str
        Names of the predictor columns.
    y_name : str
        Name of the selected target column.

    Returns
    -------
    dict of str
        Mapping from column name to ``'regression'`` or ``'classification'``.
    """
    objectives: dict[str, str] = {}
    feature_type_indices = {
        "regression": config["X_CONTINUOUS_IND"],
        "classification": [
            *config["X_BINARY_IND"],
            *config["X_MULTICAT_IND"],
        ],
    }

    # Validate that every feature index belongs to exactly one declared type.
    for objective, indices in feature_type_indices.items():
        for index in indices:
            if index < 0 or index >= len(x_columns):
                raise ValueError(f"Feature index {index} is outside the X columns.")
            column = x_columns[index]
            if column in objectives:
                raise ValueError(f"Feature {column!r} has multiple declared types.")
            objectives[column] = objective

    missing_feature_indices = set(range(len(x_columns))) - {
        index for indices in feature_type_indices.values() for index in indices
    }
    if missing_feature_indices:
        raise ValueError(
            "Feature indices missing from X_CONTINUOUS_IND, X_BINARY_IND, "
            f"or X_MULTICAT_IND: {sorted(missing_feature_indices)}"
        )

    # Y indices refer to the original Y_NAMES list, not the one-column slice.
    target_index = config["Y_NAMES"].index(y_name)
    target_types = {
        "regression": config["Y_CONTINUOUS_IND"],
        "classification": [
            *config["Y_BINARY_IND"],
            *config["Y_MULTICLASS_IND"],
        ],
    }
    matching_objectives = [
        objective
        for objective, indices in target_types.items()
        if target_index in indices
    ]
    if len(matching_objectives) != 1:
        raise ValueError(
            f"Target {y_name!r} (index {target_index}) must have exactly one "
            "declared objective in the Y_*_IND configuration keys."
        )

    if y_name in objectives:
        raise ValueError(f"Target name {y_name!r} duplicates a predictor column name.")
    objectives[y_name] = matching_objectives[0]
    return objectives


def _make_tabicl_estimator(objective: str) -> TabICLClassifier | TabICLRegressor:
    """Create the TabICL estimator corresponding to a prediction objective.

    Parameters
    ----------
    objective : str
        Either ``'regression'`` or ``'classification'``.

    Returns
    -------
    TabICLClassifier or TabICLRegressor
        A reproducibly configured TabICL estimator.
    """
    common_parameters = {
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
    }
    if objective == "regression":
        return TabICLRegressor(
            model_path="../models/tabicl-regressor-v2-20260212.ckpt",
            **common_parameters,
        )
    if objective == "classification":
        return TabICLClassifier(
            model_path="../models/tabicl-classifier-v2-20260212.ckpt",
            **common_parameters,
        )
    raise ValueError(
        "objective must be either 'regression' or 'classification'. "
        f"Received {objective!r}."
    )


def compute_bivariate_predictions(
    x: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    objective: str,
) -> float:
    """Estimate a pairwise prediction score with repeated group CV and TabICL.

    Regression is evaluated using :math:`R^2`; classification is evaluated using
    adjusted balanced accuracy. Negative fold-averaged scores are clipped to zero
    so that the returned score can be displayed on a zero-to-one heatmap.

    Parameters
    ----------
    x : pandas.DataFrame
        One-column DataFrame containing the predictor.
    y : pandas.DataFrame
        One-column DataFrame containing the prediction target.
    groups : pandas.Series
        Group labels used to keep groups separate across CV folds.
    objective : str
        Either ``'regression'`` or ``'classification'``.

    Returns
    -------
    float
        Mean cross-validated prediction score, clipped to the interval [0, 1].
    """
    if objective not in {"regression", "classification"}:
        raise ValueError(f"Unknown prediction objective: {objective!r}.")
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError("Bivariate prediction requires one predictor and one target.")

    # All three inputs must be complete for a row to enter group cross-validation.
    aligned_data = pd.concat(
        [
            x.rename(columns={x.columns[0]: "predictor"}),
            y.rename(columns={y.columns[0]: "target"}),
            groups.rename("group"),
        ],
        axis=1,
    ).dropna()
    aligned_data = aligned_data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(aligned_data) < N_CV_SPLITS + 1:
        return 0.0
    if aligned_data["group"].nunique() < N_CV_SPLITS:
        logging.warning(
            "Pairwise prediction skipped: fewer than %d unique groups.",
            N_CV_SPLITS,
        )
        return 0.0

    x_values = aligned_data[["predictor"]].to_numpy(dtype=np.float32)
    y_values = aligned_data["target"].to_numpy()
    group_values = aligned_data["group"].to_numpy()

    if objective == "classification" and pd.Series(y_values).nunique() < 2:
        return 0.0

    cross_validator = RepeatedGroupKFold(
        n_splits=N_CV_SPLITS,
        n_repeats=N_CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    estimator = _make_tabicl_estimator(objective)
    fold_scores = []

    # Each split keeps all observations from a group in either train or test.
    for train_indices, test_indices in cross_validator.split(
        x_values, groups=group_values
    ):
        x_train = x_values[train_indices]
        x_test = x_values[test_indices]
        y_train = y_values[train_indices]
        y_test = y_values[test_indices]

        if len(y_test) < 2:
            continue
        if objective == "classification":
            # A one-class training fold cannot define a useful classifier.
            if pd.Series(y_train).nunique() < 2:
                continue
            if pd.Series(y_test).nunique() < 2:
                continue

        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)

        if objective == "regression":
            score = r2_score(y_test, predictions)
        else:
            # A prediction can contain a class absent from the test fold. Such
            # folds are still scored by sklearn, but the warning is not actionable.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="y_pred contains classes not in y_true",
                    category=UserWarning,
                )
                score = balanced_accuracy_score(
                    y_test,
                    predictions,
                    adjusted=True,
                )

        if np.isfinite(score):
            fold_scores.append(float(score))

    if not fold_scores:
        return 0.0
    return float(np.clip(np.mean(fold_scores), 0.0, 1.0))


def plot_bivariate_predictions(
    config: dict[str, Any], y_name: str, store_path: str | Path
) -> None:
    """Plot TabICL cross-validated prediction scores for every feature pair.

    Rows of the matrix are predictors and columns are prediction targets. The
    diagonal is fixed to one because each variable predicts itself.

    Parameters
    ----------
    config : dict
        Analysis configuration and loaded data.
    y_name : str
        Target column included in the pairwise analysis.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    logging.info("[+] Computing pairwise predictions with TabICL.")
    x_data, y_data, group_data = get_data(
        config,
        y_name,
        drop_target_na=False,
    )
    data = pd.concat([x_data, y_data], axis=1)
    feature_names = list(data.columns)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Predictor and target column names must be unique.")

    objectives = _get_column_objectives(config, list(x_data.columns), y_name)
    pairwise_scores = np.eye(len(feature_names), dtype=float)

    # Predict each column from each other column. The matrix is directional:
    # predicting A from B need not yield the same score as predicting B from A.
    for predictor_index, predictor_name in enumerate(feature_names):
        for target_index, target_name in enumerate(feature_names):
            start_time = time.time()
            pairwise_scores[predictor_index, target_index] = (
                compute_bivariate_predictions(
                    x=data[[predictor_name]],
                    y=data[[target_name]],
                    groups=group_data.iloc[:, 0],
                    objective=objectives[target_name],
                )
            )
            logging.info(
                f"-> TabICL prediction: {predictor_name} -> {target_name}, run time: {time.time() - start_time:.2f}s"
            )

    _format_heatmap(
        matrix=pairwise_scores,
        labels=feature_names,
        title=(
            f"{config['ANALYSIS_NAME']}__{y_name}\n"
            "Bivariate Predictions via TabICL\n"
            "Rows: predictors | Columns: predictions"
        ),
        colorbar_label="Prediction score (0 to 1)",
        output_path=Path(store_path) / "4_bivariate_predictions.png",
    )

    logging.info("[✓] Computing pairwise predictions with TabICL finished.")


def plot_pca(config: dict[str, Any], y_name: str, store_path: str | Path) -> None:
    """Plot the variance explained by principal components of the feature data.

    PCA is applied to complete feature rows after standardization. The selected
    target is used to define the analysis sample but is not included in PCA.

    Parameters
    ----------
    config : dict
        Analysis configuration and loaded data.
    y_name : str
        Target column used to select the analysis sample.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    logging.info("[+] Computing principal component analysis.")
    x_data, _, _ = get_data(config, y_name)
    x_complete = x_data.replace([np.inf, -np.inf], np.nan).dropna()

    if x_complete.shape[1] == 0:
        raise ValueError("PCA requires at least one feature.")
    if len(x_complete) < 2:
        raise ValueError("PCA requires at least two complete observations.")

    # PCA cannot have more components than either samples or features.
    n_components = min(x_complete.shape[0], x_complete.shape[1])
    scaled_features = StandardScaler().fit_transform(x_complete)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(scaled_features)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    figure_width = min(1 + len(x_data.columns) * 0.6, 16)
    fig, ax = plt.subplots(figsize=(figure_width, 4))
    components = np.arange(1, len(explained_variance) + 1)

    ax.plot(
        components,
        explained_variance,
        color="cornflowerblue",
        label="Explained variance per component",
    )
    ax.plot(components, explained_variance, color="black", marker=".", linestyle="None")
    ax.set_ylim(-0.01, 1.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance per component", color="cornflowerblue")

    cumulative_ax = ax.twinx()
    cumulative_ax.plot(
        components,
        cumulative_variance,
        color="orange",
        label="Cumulative explained variance",
    )
    cumulative_ax.plot(
        components,
        cumulative_variance,
        color="black",
        marker=".",
        linestyle="None",
    )
    cumulative_ax.set_ylim(-0.01, 1.01)
    cumulative_ax.spines["top"].set_visible(False)
    cumulative_ax.spines["left"].set_visible(False)
    cumulative_ax.set_ylabel("Cumulative explained variance", color="orange")

    # Annotate both curves with rounded values for quick visual interpretation.
    for component, value in zip(components, explained_variance):
        ax.annotate(f"{value:.2f}", (component, value), fontsize=8)
    for component, value in zip(components, cumulative_variance):
        cumulative_ax.annotate(f"{value:.2f}", (component, value), fontsize=8)

    ax.set_title(
        f"{config['ANALYSIS_NAME']}__{y_name}\n"
        "Dominant patterns of variation and structure in the data via PCA\n",
        fontsize=10,
    )
    _save_and_show_figure(fig, Path(store_path) / "5_pca.png")

    logging.info("[✓] Computing principal component analysis finished.")


def _get_categorical_feature_indices(config: dict[str, Any]) -> list[int]:
    """Return the configured categorical feature indices for TabICL.

    Parameters
    ----------
    config : dict
        Analysis configuration containing ``X_BINARY_IND`` and
        ``X_MULTICAT_IND``.

    Returns
    -------
    list of int
        Sorted, unique zero-based indices of categorical predictor columns.
    """
    categorical_indices = sorted(
        set(config["X_BINARY_IND"]) | set(config["X_MULTICAT_IND"])
    )
    n_features = len(config["X_NAMES"])
    invalid_indices = [
        index for index in categorical_indices if index < 0 or index >= n_features
    ]
    if invalid_indices:
        raise ValueError(
            f"Categorical feature indices outside the X columns: {invalid_indices}"
        )
    return categorical_indices


def plot_outlier_detection(
    config: dict[str, Any],
    y_name: str,
    store_path: str | Path,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = N_ESTIMATORS,
    n_permutations: int = N_OUTLIER_PERMUTATIONS,
) -> None:
    """Detect unusual observations with TabICLUnsupervised outlier detection.

    The method estimates each observation's joint feature density using
    ``TabICLUnsupervised.score_samples``. Lower scores indicate observations that
    are less typical under the learned joint distribution. KDE smoothing is applied
    to the joint density scores, a robust normal distribution is fitted where the
    location (mean) is aligned with the center of the KDE peak/plateau and the scale
    is derived from the Full Width at Half Maximum (FWHM) of the KDE curve, and the
    cutoff is derived from this fitted normal distribution using the specified
    contamination level (default 0.01).

    Rows with missing or infinite feature values are excluded from scoring and
    retained in the exported table with missing scores and flags.

    Parameters
    ----------
    config : dict
        Analysis configuration and loaded data.
    y_name : str
        Target column used to select the analysis sample.
    store_path : str or pathlib.Path
        Directory in which the plot and Excel output are saved.
    contamination : float, default=0.01
        Fraction of the lowest-scoring observations to flag based on the robust
        fitted normal distribution. Must be between zero and 0.5.
    n_estimators : int, default=8
        Number of ensemble estimators used by TabICLUnsupervised.
    n_permutations : int, default=10
        Number of feature orderings used to estimate each joint-density score.

    Returns
    -------
    None
    """
    logging.info("[+] Detecting outliers with TabICLUnsupervised.")
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer.")

    x_data, _, _ = get_data(config, y_name)
    valid_rows = x_data.notna().all(axis=1) & np.isfinite(x_data).all(axis=1)
    x_complete = x_data.loc[valid_rows].copy()

    if len(x_complete) < 10:
        raise ValueError(
            "At least 10 complete observations are required for outlier detection."
        )
    if x_complete.shape[1] < 1:
        raise ValueError("At least one feature is required for outlier detection.")

    # TabICLUnsupervised uses these indices to choose classifier or regressor
    # conditionals. This avoids relying on automatic cardinality-based detection.
    categorical_indices = _get_categorical_feature_indices(config)
    model = TabICLUnsupervised(
        n_estimators=n_estimators,
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE,
    )
    model.fit(x_complete.to_numpy())
    scores = model.score_samples(x_complete.to_numpy(), n_permutations=n_permutations)
    scores_log = np.log10(scores)

    # KDE smoothing
    kde = gaussian_kde(scores_log)

    # Grid for evaluating KDE
    x_grid = np.linspace(scores_log.min(), scores_log.max(), 2000)
    kde_densities = kde(x_grid)
    max_dens = np.max(kde_densities)

    # Align normal distribution peak (mu) stably with the center of the KDE peak/plateau region
    # (using the median of all grid points where KDE density is >= 80% of the maximum density).
    peak_mask = kde_densities >= 0.80 * max_dens
    mu = float(np.median(x_grid[peak_mask]))

    # Determine spread (std) from the Full Width at Half Maximum (FWHM) of the KDE curve
    # This matches the normal distribution's spread directly to the KDE peak's width while ignoring tail edges.
    half_max_mask = kde_densities >= 0.50 * max_dens
    x_half_max = x_grid[half_max_mask]
    fwhm = x_half_max[-1] - x_half_max[0]
    std = (
        fwhm / 2.35482
    )  # Factor relating FWHM to standard deviation of a normal distribution
    if std == 0.0 or not np.isfinite(std):
        std = float(np.std(scores_log)) or 1.0

    # Thresholds for outlier
    score_threshold_log = norm.ppf(contamination, loc=mu, scale=std)
    score_threshold = 10**score_threshold_log
    is_outlier = scores <= score_threshold
    score_percentiles = pd.Series(scores, index=x_complete.index).rank(
        method="average",
        pct=True,
    )

    # Preserve all original observations in the output for straightforward joins.
    result_data = pd.DataFrame(index=x_data.index)
    result_data["outlier_score"] = np.nan
    result_data["is_outlier"] = pd.Series(
        pd.NA,
        index=x_data.index,
        dtype="boolean",
    )
    result_data["outlier_score_rank_percentile"] = np.nan
    result_data.loc[x_complete.index, "outlier_score"] = scores
    result_data.loc[x_complete.index, "is_outlier"] = is_outlier
    result_data.loc[x_complete.index, "outlier_score_rank_percentile"] = (
        score_percentiles
    )
    result_data.index.name = x_data.index.name or "observation_index"

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the histogram bars without KDE
    sns.histplot(
        x=scores_log,
        bins=31,
        kde=False,
        color="#777777",
        edgecolor="#444444",
        linewidth=0.8,
        ax=ax,
    )

    # Plot the KDE line scaled to the frequency/count axis
    xmin, xmax = ax.get_xlim()
    x_axis = np.linspace(xmin, xmax, 200)
    bin_width = (xmax - xmin) / 31
    kde_pdf = kde(x_axis) * len(scores_log) * bin_width
    ax.plot(
        x_axis,
        kde_pdf,
        color="#555555",
        linewidth=1.5,
        label="KDE smoothed joint density",
    )

    # Add the robustly fitted normal distribution curve
    normal_pdf = norm.pdf(x_axis, loc=mu, scale=std) * len(scores_log) * bin_width
    ax.plot(
        x_axis,
        normal_pdf,
        color="crimson",
        linestyle="-",
        linewidth=1.5,
        label="Robust Normal",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("TabICL joint-density score (log10, lower = more anomalous)")
    ax.set_ylabel("Count")
    ticks = np.geomspace(scores.min(), scores.max(), 6)
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"{t:.3g}" for t in ticks])
    ax.axvline(
        score_threshold_log,
        color="black",
        linestyle="--",
        label=f"Robust Normal cutoff ({contamination:.1%})",
    )
    ax.legend(frameon=False, loc="upper right")

    flagged_percentage = float(is_outlier.mean() * 100)
    n_excluded = int((~valid_rows).sum())
    ax.set_title(
        f"{config['ANALYSIS_NAME']}__{y_name}\n"
        "Outlier detection via TabICLUnsupervised: "
        f"{flagged_percentage:.1f}% flagged "
        f"({n_excluded} incomplete rows excluded)\n",
        fontsize=10,
    )

    output_dir = Path(store_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_data.to_excel(
        output_dir / "6_tabicl_outlier_detection.xlsx",
        index=True,
    )
    _save_and_show_figure(fig, output_dir / "6_tabicl_outlier_detection.png")

    logging.info(
        "[✓] Outlier detection finished: %d/%d complete observations flagged; "
        "%d rows excluded due to incomplete features.",
        int(is_outlier.sum()),
        len(x_complete),
        n_excluded,
    )


def main() -> None:
    """Run all exploratory analyses for every configured target.

    Returns
    -------
    None
    """
    setup_logging()
    document_requirements()

    config_names = get_config_names(CONFIG_NAMES_PATH)
    configs = get_configs(get_config_file_paths(config_names))

    for config in configs:
        for y_name in config["Y_NAMES"]:
            analysis_name = f"{config['ANALYSIS_NAME']}__{y_name}"
            logging.info("Starting exploratory data analysis: %s", analysis_name)
            output_dir = DATA_OUTPUT_DIR / analysis_name
            output_dir.mkdir(parents=True, exist_ok=True)

            plot_1d_distribution(config, y_name, output_dir)
            plot_2d_distribution(config, y_name, output_dir)
            plot_bivariate_correlation(config, y_name, output_dir)
            plot_bivariate_predictions(config, y_name, output_dir)
            plot_pca(config, y_name, output_dir)
            plot_outlier_detection(config, y_name, output_dir)

    logging.info("[✓] Exploratory data analysis finished.")


if __name__ == "__main__":
    main()
