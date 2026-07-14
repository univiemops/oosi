# -*- coding: utf-8 -*-
"""Out-of-sample inference 3 - Exploratory Data Analysis
v001

@author: david.steyrl@univie.ac.at
"""

import importlib.metadata as im
import logging
import os
import platform
from typing import Any, Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score, r2_score
from tabicl import TabICLClassifier
from tabicl import TabICLRegressor

from custom_splitter import RepeatedGroupKFold

# Suppress warnings that clog stdout/stderr during rapid solver iterations
WARNING_MSGS = ["y_pred contains classes not in y_true"]
for msg in WARNING_MSGS:
    os.environ["PYTHONWARNINGS"] = f"ignore:{msg}:::"
    warnings.filterwarnings("ignore", message=msg)


def setup_logging() -> None:
    """Configure simultaneous logging to both a project file and the terminal."""
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/oosi_3_eda.log",
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

    logging.info("Logging established")
    logging.info(f"Logs save at: {log_dir}/oosi_3_eda.log")


def document_requirements() -> None:
    """Document environmental dependencies, logging system configuration and Python specs."""
    # Get distributions
    lines = [f"{dist.name}=={dist.version}" for dist in im.distributions()]

    # Write requirements
    with open("../logs/requirements_oosi_3_eda.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    logging.info(f"OS: {platform.system()} {platform.release()}")
    logging.info(f"Python Version: {platform.python_version()}")
    logging.info("Dependencies save at: ../logs/requirements_oosi_3_eda.txt")


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
    config: Dict[str, Any], y_name: str, drop_target_na=True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrieve features, target, and groups aligned on non-null targets."""
    if drop_target_na:
        y_df = config["Y"][y_name].to_frame().dropna()
    elif not drop_target_na:
        y_df = config["Y"][y_name].to_frame()
    g_df = config["G"].reindex(index=y_df.index)
    x_df = config["X"].reindex(index=y_df.index)

    logging.info(f"Data sliced for {y_name}.")
    dropped = config["Y"].shape[0] - y_df.shape[0]
    if dropped > 0:
        logging.info(f"{dropped} samples dropped due to NaNs in {y_name}.")

    return x_df, y_df, g_df


def plot_1D_distribution(config: dict, y_name: str, store_path: str) -> None:
    """Plot 1D data distributions"""
    logging.info("1D Data Distributions.")

    x_df, y_df, g_df = get_data(config, y_name)
    data = pd.concat([x_df, y_df], axis=1)

    # Figure
    names_max_len = max(len(name) for name in data.columns)
    n_names = len(data.columns)
    fig_size = (
        names_max_len * 0.1 + 5,
        (n_names + 1) * 1.1 + 1,
    )
    fig, axes = plt.subplots(nrows=n_names, ncols=1, figsize=fig_size, sharex=False)

    # Violin plot
    for ax, column in zip(axes, data.columns):
        sns.violinplot(
            data=data[column],
            bw_method="scott",  # Bandwidth estimation method
            bw_adjust=0.3,  # Adjust bandwidth
            cut=2,  # Extend density beyond data
            density_norm="width",  # Normalize density by width
            gridsize=100,  # Number of points in density estimation
            width=0.8,  # Width of violin plot
            inner="box",  # Show box plot inside violins
            orient="h",  # Horizontal orientation
            linewidth=1,  # Line width of violin edges
            color="#777777",  # Violin color
            saturation=1.0,  # Saturation level
            ax=ax,  # Axes to plot on
        )

        # Formatting subplot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(
            column, rotation="horizontal", horizontalalignment="right", fontsize=10
        )
        ax.set_xlabel("", fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)

    # Formatting figure
    ax.set_xlabel("Range", fontsize=10)
    title_str = f"{config['ANALYSIS_NAME']}__{y_name}\nData Distributions (1D)"
    plt.suptitle(title_str, fontsize=10, y=0.94)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save
    save_base_path = f"{store_path}/1_distri_1D"
    png_save_path = f"{save_base_path}.png"
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_2D_distribution(config: dict, y_name: str, store_path: str) -> None:
    """Plot 2D data distributions"""
    logging.info("2D Data Distributions.")

    x_df, y_df, g_df = get_data(config, y_name)
    data = pd.concat([x_df, y_df], axis=1)

    # Pairplot
    pair_plot = sns.pairplot(
        data,
        corner=False,  # Include all pairwise plots
        diag_kind="kde",  # Use Kernel Density Estimate (KDE) for diagonal
        plot_kws={"color": "#777777"},  # Set color for pairwise plots
        diag_kws={"color": "#777777"},  # Set color for diagonal plots
    )

    # Formatting figure
    title_str = f"{config['ANALYSIS_NAME']}__{y_name}\nData Distributions (2D)"
    pair_plot.fig.suptitle(title_str, fontsize=10, y=1.0)
    pair_plot.map_lower(sns.kdeplot, levels=3, color=".2")

    # Save
    save_base_path = f"{store_path}/2_distri_2D"
    png_save_path = f"{save_base_path}.png"
    pair_plot.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_joint_linear(config: dict, y_name: str, store_path: str) -> None:
    """Plot Linear Joint Information data (Correlation Heatmap)"""
    logging.info("Linear Joint Information (Correlation Heatmap).")

    x_df, y_df, g_df = get_data(config, y_name)
    data = pd.concat([x_df, y_df], axis=1)

    # Figure
    names_max_len = max(len(name) for name in data.columns)
    n_names = len(data.columns)
    fig_size = (
        n_names * 0.6 + names_max_len * 0.1 + 1,
        n_names * 0.6 + names_max_len * 0.1 + 1,
    )
    fig, ax = plt.subplots(figsize=fig_size)
    colorbar_label = "|Correlation| (0 to 1)"

    # Correlation heatmap
    sns.heatmap(
        data.corr().abs(),  # Compute correlations
        vmin=0,
        vmax=1,  # Value range
        cmap="Greys",  # Color map
        robust=True,
        annot=True,  # Show correlation values
        fmt=".2f",  # Format for annotations
        annot_kws={"size": 10},  # Annotation font size
        linewidths=1,  # Grid line width
        linecolor="#999999",  # Grid line color
        cbar=True,  # Show color bar
        cbar_kws={"label": colorbar_label, "shrink": 0.6},  # Color bar settings
        square=True,  # Square cells
        xticklabels=data.columns,
        yticklabels=data.columns,
        ax=ax,
    )

    # Formatting figure
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    title_str = f"{config['ANALYSIS_NAME']}__{y_name}\nJoint Information in Data (Linear, Correlation)\n"
    plt.title(title_str, fontsize=10)
    cb_ax = fig.axes[1]
    cb_ax.tick_params(labelsize=10)
    cb_ax.set_ylabel(colorbar_label, fontsize=10)
    cb_ax.set_box_aspect(50)

    # Save
    save_base_path = f"{store_path}/3_joint_linear"
    png_save_path = f"{save_base_path}.png"
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def compute_pair_predictions(
    x: pd.DataFrame, y: pd.DataFrame, g: pd.Series, objective: str
) -> float:
    """Compute pairwise prediction score using TabICL.
    R² for regression, adjusted balanced accuracy for classification.

    Parameters
    ----------
    x: pd.DataFrame
        DataFrame holding the single predictor feature.
    y: pd.DataFrame
        DataFrame holding the target feature.
    g: pd.Series
        Series holding group data for cross-validation.
    objective: str
        Objective type, either "regression" or "classification".

    Returns
    -------
    float: Pairwise prediction score (>=0).
    """
    # Remove Nans
    y.dropna(inplace=True)
    x_na = x.reindex(index=y.index)
    x_na.dropna(inplace=True)
    y_na = y.reindex(index=x_na.index)
    g_na = g.reindex(index=x_na.index)

    x_na.reset_index(inplace=True, drop=True)
    y_na.reset_index(inplace=True, drop=True)

    cv = RepeatedGroupKFold(n_splits=5, n_repeats=2, random_state=None)
    scores = []
    if objective == "continuous":
        estimator = TabICLRegressor(
            kv_cache=True,
            model_path="../models/tabicl-regressor-v2-20260212.ckpt",
            random_state=None,
        )
    elif objective == "categorical":
        estimator = TabICLClassifier(
            kv_cache=True if y_na.iloc[:, 0].nunique() <= 10 else False,
            model_path="../models/tabicl-classifier-v2-20260212.ckpt",
            random_state=None,
        )

    # Cross-validation loop
    for i_cv, (i_trn, i_tst) in enumerate(cv.split(g_na, groups=g_na)):
        y_trn = y_na.iloc[i_trn]
        x_trn = x_na.iloc[i_trn]
        y_tst = y_na.iloc[i_tst]
        x_tst = x_na.iloc[i_tst]

        # Robust fit and score prediction
        estimator.fit(x_trn.to_numpy(), y_trn.to_numpy().squeeze())
        y_pred = estimator.predict(x_tst.to_numpy().squeeze().reshape(-1, 1))

        if objective == "continuous":
            score = r2_score(y_tst.to_numpy().squeeze(), y_pred)
        elif objective == "categorical":
            score = balanced_accuracy_score(
                y_tst.to_numpy().squeeze(), y_pred, adjusted=True
            )
        else:
            raise ValueError(f"Objective is {objective}.")

        scores.append(score)

        pair_pred = max(0.0, float(np.nanmean(scores)))

    return pair_pred


def plot_pair_predictions(config: dict, y_name: str, store_path: str) -> None:
    """Pairwise Prediction Heatmap"""
    logging.info("Pairwise Prediction Heatmap via TabICL.")

    x_df, y_df, g_df = get_data(config, y_name, drop_target_na=False)
    data = pd.concat([x_df, y_df], axis=1)
    feature_names = data.columns
    feature_count = len(feature_names)

    # Initialize pairwise prediction matrix (diagonal starts as 1)
    pair_predictions = np.ones((feature_count, feature_count))

    # Pre-compute the task objective for each feature to prevent redundant evaluations
    column_objectives = {}
    for idx, col in enumerate(x_df.columns):
        if idx in config["X_CONTINUOUS_IND"]:
            column_objectives[col] = "continuous"
        elif idx in config["X_BINARY_IND"]:
            column_objectives[col] = "categorical"
        elif idx in config["X_MULTICAT_IND"]:
            column_objectives[col] = "categorical"
        else:
            raise ValueError(f"Index {idx} not in X_IND.")

    for idx, col in enumerate(y_df.columns):
        if idx in config["Y_CONTINUOUS_IND"]:
            column_objectives[col] = "continuous"
        elif idx in config["Y_BINARY_IND"]:
            column_objectives[col] = "categorical"
        elif idx in config["Y_MULTICLASS_IND"]:
            column_objectives[col] = "categorical"
        else:
            raise ValueError(f"Index {idx} not in Y_IND.")

    # Iteratively run TabICL on all pairwise permutations
    for id_pred1, name_pred1 in enumerate(feature_names):
        for id_pred2, name_pred2 in enumerate(feature_names):
            if id_pred1 == id_pred2:
                continue  # Skip diagonal

            # Slicing safely out of pre-allocated df_all
            xt = data[[name_pred1]]
            yt = data[[name_pred2]]
            objective = column_objectives[name_pred2]

            # Evaluate pairwise predicting capability
            pair_predictions[id_pred1, id_pred2] = compute_pair_predictions(
                x=xt, y=yt, g=g_df, objective=objective
            )

    # Figure
    names_max_len = max(len(name) for name in feature_names)
    fig_size = (
        feature_count * 0.6 + names_max_len * 0.1 + 1,
        feature_count * 0.6 + names_max_len * 0.1 + 1,
    )
    fig, ax = plt.subplots(figsize=fig_size)
    colorbar_label = "|Pairwise Predictions| (0 to 1)"

    sns.heatmap(
        pair_predictions,
        vmin=0,
        vmax=1,
        cmap="Greys",
        robust=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=True,
        cbar_kws={"label": colorbar_label, "shrink": 0.6},
        square=True,
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=ax,
    )
    title_str = (
        f"{config['ANALYSIS_NAME']}__{y_name}\n"
        "Pairwise Predictions via TabICL\n"
        "Y-axis: Predictors, X-axis: Prediction Targets"
    )
    plt.title(title_str, fontsize=10)

    # Save
    save_base_path = f"{store_path}/4_pair_predictions"
    png_save_path = f"{save_base_path}.png"
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pca(config: dict, y_name: str, store_path: str) -> None:
    """Multidimensional Pattern Visualization with PCA"""
    logging.info("Multidimensional Pattern Visualization with PCA.")

    x_df, y_df, g_df = get_data(config, y_name)

    # PCA
    pca = PCA(
        n_components=x_df.shape[1],
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0001,
        iterated_power="auto",
        random_state=None,
    )
    pca.fit(x_df.dropna())

    # Figure
    n_names = len(x_df.columns)
    fig_width = min((1 + n_names * 0.6), 16)
    fig_size = (fig_width, 4)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(
        pca.explained_variance_ratio_,
        color="cornflowerblue",
        label="Explained variance per component",
    )
    ax.plot(
        pca.explained_variance_ratio_,
        color="black",
        marker=".",
        linestyle="None",
    )
    ax.set_xlim((-0.01, ax.get_xlim()[1]))
    ax.set_ylim((-0.01, 1.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("PCA-component")
    ax.set_ylabel("Explained Variance", color="cornflowerblue")
    ax2 = ax.twinx()
    ax2.plot(
        np.cumsum(pca.explained_variance_ratio_),
        color="orange",
        label="Cumulative explained variance",
    )
    ax2.plot(
        np.cumsum(pca.explained_variance_ratio_),
        color="black",
        marker=".",
        linestyle="None",
    )
    ax2.set_xlim((-0.01, ax2.get_xlim()[1]))
    ax2.set_ylim((-0.01, 1.01))
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_ylabel("Cumulative Variance", color="orange")
    for comp, t in enumerate(pca.explained_variance_ratio_.round(2)):
        ax.text(comp, t, t, fontsize=10)
    for comp, t in enumerate(np.cumsum(pca.explained_variance_ratio_).round(2)):
        ax2.text(comp, t, t, fontsize=10)
    title_str = f"{config['ANALYSIS_NAME']}__{y_name}\nMultidimensional pattern in data via PCA (linear)\n"
    plt.title(title_str, fontsize=10)

    # Save
    save_base_path = f"{store_path}/5_pca"
    png_save_path = f"{save_base_path}.png"
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_outlier_detection(config: dict, y_name: str, store_path: str) -> None:
    """Outlier Detection using Isolation Forest"""
    logging.info("Outlier Detection using Isolation Forest.")

    x_df, y_df, g_df = get_data(config, y_name)

    # Isolation Forest
    iForest = IsolationForest(
        n_estimators=10000,
        max_samples="auto",
        contamination="auto",
        max_features=0.66,
        bootstrap=False,
        n_jobs=-2,  # Use all available processors
        random_state=None,
        verbose=0,
        warm_start=False,
    )
    outlier = iForest.fit_predict(x_df)
    outlier_df = pd.DataFrame(data=outlier, columns=["is_outlier"])
    outlier_score = iForest.decision_function(x_df)

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=outlier_score, bins=30, kde=True, color="#777777", ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Isolation Forest outlier score")
    ax.set_ylabel("Count")
    outlier_percentage = np.sum(outlier == -1) / len(outlier) * 100
    title_str = f"{config['ANALYSIS_NAME']}__{y_name}\nOutlier in data via Isolation Forest: {outlier_percentage:.1f}% potential outliers"
    plt.title(title_str, fontsize=10)

    # Save
    save_base_path = f"{store_path}/6_iForest"
    excel_save_path = f"{save_base_path}.xlsx"
    outlier_df.to_excel(excel_save_path, index=False)
    png_save_path = f"{save_base_path}.png"
    plt.savefig(png_save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    """Main function of exploratory data analysis."""
    setup_logging()
    document_requirements()
    configs_names = get_configs_names("../configs/oosi_configs_names.yaml")
    configs = get_configs(get_configs_file_names(configs_names))

    for config in configs:
        for target_idx, y_name in enumerate(config["Y_NAMES"]):
            analysis_name = f"{config['ANALYSIS_NAME']}__{y_name}"
            logging.info(f"Exploratory Data Analysis started: {analysis_name}")
            save_path = f"../eda/{analysis_name}"
            os.makedirs(save_path, exist_ok=True)

            # Plots
            plot_1D_distribution(config, y_name, save_path)

            plot_2D_distribution(config, y_name, save_path)

            plot_joint_linear(config, y_name, save_path)

            plot_pair_predictions(config, y_name, save_path)

            plot_pca(config, y_name, save_path)

            plot_outlier_detection(config, y_name, save_path)

    logging.info("Exploratory Data Analysis finished.")


if __name__ == "__main__":
    main()
