# -*- coding: utf-8 -*-
"""Out-of-sample inference 2 - Plotting
v003

@author: david.steyrl@univie.ac.at
"""

from __future__ import annotations

import importlib.metadata as metadata
import logging
import pickle as pkl
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shapiq
from scipy.stats import t
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)

# Project paths are relative to the project working directory.
LOG_DIR = Path("../logs")
RESULTS_DIR = Path("../results")
PLOTS_DIR = Path("../plots")
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
    log_path = LOG_DIR / "oosi_2_plt.log"

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

    Writes the installed distributions to ``../logs/requirements_oosi_2_plt.txt``.

    Returns
    -------
    None
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    requirements_path = LOG_DIR / "requirements_oosi_2_plt.txt"

    requirements = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    )
    requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    logging.info("Operating system: %s %s", platform.system(), platform.release())
    logging.info("Python version: %s", platform.python_version())
    logging.info("Package versions saved to: %s", requirements_path)


def plot_parameter_distributions(
    results: dict[str, Any], store_path: str | Path
) -> None:
    """Plot model hyperparameter distributions as histograms.

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)
    optimized_params = {
        key: [d[key] for d in results["total_best_params"]]
        for key in results["total_best_params"][0]
    }

    for idx, (name, data) in enumerate(optimized_params.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        use_log_scale = name.endswith(("path_smooth", "alpha", "C"))

        sns.histplot(
            data=data,
            bins=30,
            kde=True,
            color="#777777",
            log_scale=use_log_scale,
            ax=ax,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(name.split("__")[-1])
        ax.set_ylabel("Count")

        if name.endswith("colsample_bytree"):
            ax.set_xlim([0.05, 1.05])
        elif name.endswith("extra_trees"):
            ax.set_xlim([-0.05, 1.05])
        elif name.endswith("path_smooth"):
            ax.set_xlim([0.1, 1000])
        elif name.endswith("alpha"):
            ax.set_xlim([0.001, 100])
        elif name.endswith("C"):
            ax.set_xlim([0.001, 100])
        elif name.endswith("l1_ratio"):
            ax.set_xlim([-0.05, 1.05])

        param_short_name = name.split("__")[-1]
        title_str = (
            f"{results['config']['ANALYSIS_NAME']} "
            f"({results['params']['pipe_name']} estimator predicting {results['params']['y_name']})\n"
            f"Parameter distribution of {param_short_name}"
        )
        ax.set_title(title_str, fontsize=10)

        save_path = store_path / f"0_{idx}_opti_param_{param_short_name}"
        fig.savefig(f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def corrected_std(differences: np.ndarray, n_tst_over_n_trn: float = 0.25) -> float:
    """Correct standard deviation using Nadeau and Bengio's approach.

    References
    ----------
    .. [1] Nadeau, C., Bengio, Y. Inference for the Generalization Error.
           Machine Learning 52, 239–281 (2003).
           https://doi.org/10.1023/A:1024068626366

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float, default=0.25
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    float
        Variance-corrected standard deviation of the set of differences.
    """
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_tst_over_n_trn)
    return float(np.sqrt(corrected_var))


def corrected_ttest(
    differences: np.ndarray, n_tst_over_n_trn: float = 0.25
) -> tuple[float, float]:
    """Compute right-tailed paired t-test with corrected variance.

    References
    ----------
    .. [1] Nadeau, C., Bengio, Y. Inference for the Generalization Error.
           Machine Learning 52, 239–281 (2003).
           https://doi.org/10.1023/A:1024068626366

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float, default=0.25
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean_diff = np.nanmean(differences)
    std_diff = max(1e-6, corrected_std(differences, n_tst_over_n_trn))
    t_stat = mean_diff / std_diff
    p_val = float(t.sf(t_stat, df=len(differences) - 1))
    return float(t_stat), p_val


def plot_regression_scatter(results: dict[str, Any], store_path: str | Path) -> None:
    """Plot regression model fit and performance metrics in a scatter plot.

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    true_values = np.concatenate(true_values_per_fold)
    pred_values = np.concatenate(pred_values_per_fold)

    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]

    mae = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    mae_sh = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    r2 = [r2_score(i, j) for i, j in zip(true_values_per_fold, pred_values_per_fold)]
    r2_sh = [
        r2_score(i, j) for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pred_values, true_values, zorder=2, alpha=0.1, color="#444444")
    ax.plot(
        [-10000, 10000],
        [-10000, 10000],
        color="#999999",
        zorder=3,
        linewidth=2,
        alpha=0.3,
    )
    ax.set_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both", which="major", bottom=True, top=False, left=True, right=False
    )
    ax.grid(visible=True, which="major", axis="both", linestyle=":", alpha=0.5)

    true_values_range = max(true_values) - min(true_values)
    margin = true_values_range / 20
    ax.set_xlim(min(true_values) - margin, max(true_values) + margin)
    ax.set_ylim(min(true_values) - margin, max(true_values) + margin)

    analysis_name = results["config"]["ANALYSIS_NAME"]
    pipe_name = results["params"]["pipe_name"]
    y_name = results["params"]["y_name"]
    ax.set_title(
        f"{analysis_name} ({pipe_name} estimator predicting {y_name})\n", fontsize=10
    )
    ax.set_xlabel(f"Predicted {y_name}", fontsize=10)
    ax.set_ylabel(f"True {y_name}", fontsize=10)

    _, pval_mae = corrected_ttest(np.array(mae_sh) - np.array(mae))
    ax.text(
        0.3,
        0.09,
        f"Original data: MAE mean±std {np.nanmean(mae):.2f}±{np.std(mae):.2f} | med {np.nanmedian(mae):.2f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.text(
        0.3,
        0.055,
        f"Shuffled data: MAE mean±std {np.nanmean(mae_sh):.2f}±{np.std(mae_sh):.2f} | med {np.nanmedian(mae_sh):.2f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    pval_mae_str = "p≤0.001" if pval_mae <= 0.001 else f"p={pval_mae:.3f}"
    ax.text(
        0.3,
        0.02,
        f"Original vs. shuffled: {pval_mae_str}",
        transform=ax.transAxes,
        fontsize=10,
    )

    _, pval_r2 = corrected_ttest(np.array(r2) - np.array(r2_sh))
    ax.text(
        0.02,
        0.96,
        f"Original data: R² mean±std {np.nanmean(r2):.3f}±{np.std(r2):.3f} | med {np.nanmedian(r2):.3f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.text(
        0.02,
        0.925,
        f"Shuffled data: R² mean±std {np.nanmean(r2_sh):.3f}±{np.std(r2_sh):.3f} | med {np.nanmedian(r2_sh):.3f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    pval_r2_str = "p≤0.001" if pval_r2 <= 0.001 else f"p={pval_r2:.3f}"
    ax.text(
        0.02,
        0.89,
        f"Original vs. shuffled: {pval_r2_str}",
        transform=ax.transAxes,
        fontsize=10,
    )

    save_path = store_path / "1_0_predictions"
    fig.savefig(f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_regression_violin(results: dict[str, Any], store_path: str | Path) -> None:
    """Plot regression metrics distributions in a violin plot.

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]

    mae = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    mae_sh = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    r2 = [r2_score(i, j) for i, j in zip(true_values_per_fold, pred_values_per_fold)]
    r2_sh = [
        r2_score(i, j) for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]

    scores_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae)),
            "R²": pd.Series(np.array(r2)),
            "Data": pd.Series(["original" for _ in mae]),
            "Dummy": pd.Series(np.ones(np.array(mae).shape).flatten()),
        }
    )
    scores_sh_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae_sh)),
            "R²": pd.Series(np.array(r2_sh)),
            "Data": pd.Series(["shuffled" for _ in mae_sh]),
            "Dummy": pd.Series(np.ones(np.array(mae_sh).shape).flatten()),
        }
    )
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    metrics = ["Mean Absolute Error", "R²"]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics) * 1 + 1))
    fig.tight_layout()
    mypal = {"original": "#777777", "shuffled": "#eeeeee"}

    for i, metric in enumerate(metrics):
        sns.violinplot(
            x=metric,
            y="Dummy",
            hue="Data",
            data=all_scores_df,
            bw_method="scott",
            bw_adjust=0.5,
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=1,
            saturation=1,
            ax=axes[i],
            palette=mypal,
        )
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["left"].set_visible(False)
        axes[i].tick_params(
            axis="both",
            which="major",
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        axes[i].set_xlabel(metric, fontsize=10)
        axes[i].set_ylabel("", fontsize=10)
        if i > 0 and axes[i].legend() is not None:
            axes[i].legend().remove()
        axes[i].set_axisbelow(True)
        axes[i].grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)

    analysis_name = results["config"]["ANALYSIS_NAME"]
    pipe_name = results["params"]["pipe_name"]
    y_name = results["params"]["y_name"]
    title_str = f"{analysis_name} ({pipe_name} estimator predicting {y_name})\n"
    axes[0].set_title(title_str, fontsize=10)

    save_path = store_path / "1_1_predictions_distribution"
    fig.savefig(f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_classification_confusion(
    results: dict[str, Any], store_path: str | Path
) -> None:
    """Plot classification confusion matrices.

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    true_values = np.concatenate(true_values_per_fold)
    pred_values = np.concatenate(pred_values_per_fold)  # noqa

    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]

    acc = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    acc_sh = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    class_labels = np.unique(true_values)

    con_mat_count = None
    for true, pred in zip(true_values_per_fold, pred_values_per_fold):
        cm = confusion_matrix(true, pred, labels=class_labels, normalize=None)
        con_mat_count = cm if con_mat_count is None else np.add(con_mat_count, cm)

    con_mat = None
    for true, pred in zip(true_values_per_fold, pred_values_per_fold):
        cm = confusion_matrix(true, pred, labels=class_labels, normalize="true")
        con_mat = cm if con_mat is None else np.add(con_mat, cm)

    con_mat_norm = con_mat / len(true_values_per_fold)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(con_mat.shape[0] * 0.5 + 4, con_mat.shape[0] * 0.5 + 3.5),
    )
    fig.tight_layout()

    y_name = results["params"]["y_name"]
    sns.heatmap(
        con_mat_count,
        cmap="Greys",
        robust=True,
        annot=True,
        fmt="",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        ax=axes[0],
    )
    axes[0].set_title("num. predictions", fontsize=10)
    axes[0].set_xlabel(f"Predicted {y_name}", fontsize=10)
    axes[0].set_ylabel(f"True {y_name}", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=10, labelrotation=0)

    sns.heatmap(
        con_mat_norm * 100,
        cmap="Greys",
        robust=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        ax=axes[1],
    )
    axes[1].set_title(f"norm. to True {y_name}", fontsize=10)
    axes[1].set_xlabel(f"Predicted {y_name}", fontsize=10)
    axes[1].set_ylabel(f"True {y_name}", fontsize=10)
    axes[1].tick_params(axis="y", labelsize=10, labelrotation=0)

    _, pval_acc = corrected_ttest(np.array(acc) - np.array(acc_sh))
    pval_acc_str = "p≤0.001" if pval_acc <= 0.001 else f"p={pval_acc:.3f}"

    analysis_name = results["config"]["ANALYSIS_NAME"]
    pipe_name = results["params"]["pipe_name"]
    title_str = (
        f"{analysis_name} ({pipe_name} estimator predicting {y_name})\n"
        f"Original data acc: mean±std {np.nanmean(acc):.3f}±{np.std(acc):.3f} | med {np.nanmedian(acc):.3f}\n"
        f"Shuffled data acc: mean±std {np.nanmean(acc_sh):.3f}±{np.std(acc_sh):.3f} | med {np.nanmedian(acc_sh):.3f}\n"
        f"Original vs. shuffled: {pval_acc_str}"
    )
    fig.suptitle(title_str, fontsize=10, y=0.95)

    save_path = store_path / "1_0_predictions"
    fig.savefig(f"{str(save_path)}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_classification_violin(results: dict[str, Any], store_path: str | Path) -> None:
    """Plot classification accuracy distributions in a violin plot.

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]

    acc = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    acc_sh = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]

    scores_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc)),
            "Data": pd.Series(["original" for _ in acc]),
            "Dummy": pd.Series(np.ones(np.array(acc).shape).flatten()),
        }
    )
    scores_sh_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc_sh)),
            "Data": pd.Series(["shuffled" for _ in acc_sh]),
            "Dummy": pd.Series(np.ones(np.array(acc_sh).shape).flatten()),
        }
    )
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)

    fig, ax = plt.subplots(figsize=(8, 3))
    mypal = {"original": "#777777", "shuffled": "#eeeeee"}

    sns.violinplot(
        x="Accuracy",
        y="Dummy",
        hue="Data",
        data=all_scores_df,
        bw_method="scott",
        bw_adjust=0.5,
        cut=2,
        density_norm="width",
        gridsize=100,
        width=0.8,
        inner="box",
        orient="h",
        linewidth=1,
        saturation=1,
        ax=ax,
        palette=mypal,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        top=False,
        left=False,
        right=False,
        labelleft=False,
    )
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.set_ylabel("", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)

    analysis_name = results["config"]["ANALYSIS_NAME"]
    pipe_name = results["params"]["pipe_name"]
    y_name = results["params"]["y_name"]
    title_str = (
        f"{analysis_name} ({pipe_name} estimator predicting {y_name})\n"
        f"Predicting {y_name}"
    )
    ax.set_title(title_str, fontsize=10)

    save_path = store_path / "1_1_predictions_distribution"
    fig.savefig(f"{str(save_path)}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_avg_shap_values(results: dict[str, Any], store_path: str | Path) -> None:
    """Plot average absolute SHAP values (global effects).

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)

    def _calculate_mean_abs_shap(
        folds_data: list, target_class: str
    ) -> tuple[dict, dict]:
        all_keys = {
            key
            for fold in folds_data
            for item in fold.get(target_class, [])
            for key in item.dict_values.keys()
        }
        fold_means = defaultdict(list)
        for fold in folds_data:
            class_fold = fold.get(target_class, [])
            for key in all_keys:
                abs_vals = [np.abs(item.dict_values.get(key, 0)) for item in class_fold]
                if abs_vals:
                    fold_means[key].append(np.nanmean(abs_vals))
                else:
                    fold_means[key].append(0.0)
        return {key: np.nanmean(vals) for key, vals in fold_means.items()}, fold_means

    x_names = results["config"]["X_NAMES"]
    params = results["params"]
    format_key = lambda key: " x ".join(str(x_names[idx]) for idx in key)

    for c_class in params["classes"]:
        shap_mean_abs, shap_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii"], c_class
        )
        shap_sh_mean_abs, shap_sh_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii_sh"], c_class
        )

        shapii_base = shap_mean_abs.pop((), 0.0)
        shap_sh_mean_abs.pop((), None)

        shapii_se_sort = pd.Series(
            {format_key(k): v for k, v in shap_mean_abs.items()}
        ).sort_values(ascending=True)

        pval_dict = {}
        for key in shap_folds_dct.keys():
            if key == ():
                continue
            diff = np.array(shap_folds_dct[key]) - np.array(shap_sh_folds_dct[key])
            _, c_pval = corrected_ttest(diff)
            pval_dict[format_key(key)] = c_pval

        pval_se = pd.Series(pval_dict)

        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
            f"mean(|SHAP values|): average effect on {params['y_name']}\n"
            f"mean(|SHAP values|): average change from expected value of {np.round(shapii_base, decimals=2)}"
        )
        if params["target_type"] in ["binary", "multiclass"]:
            title_str += f" (logit)\n class: {c_class}"
        title_lines_count = title_str.count("\n") + 1
        x_names_max_len = (
            max(len(str(i)) for i in shapii_se_sort.index)
            if not shapii_se_sort.empty
            else 10
        )

        figsize = (
            x_names_max_len * 0.1 + 8,
            len(shapii_se_sort) * 0.4 + title_lines_count * 0.4 + 0.5,
        )
        fig, ax = plt.subplots(figsize=figsize)
        shapii_se_sort.plot(kind="barh", color="#777777", fontsize=10, ax=ax)

        ax.set_xlabel("mean(|SHAP values|)", fontsize=10)
        ax.set_title(title_str, fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)

        max_val = shapii_se_sort.max() if not shapii_se_sort.empty else 1.0
        for i, (c_pred, c_val) in enumerate(shapii_se_sort.items()):
            p_val = pval_se.get(c_pred, 1.0)
            pval_string = "p≤0.001" if p_val <= 0.001 else f"p={p_val:.3f}"
            txt_str = f"{c_val:.2f} | {pval_string}"
            ax.text(
                c_val + max_val * 0.01,
                i,
                txt_str,
                color="k",
                va="center",
                fontsize=10,
            )

        x_left, x_right = ax.get_xlim()
        ax.set_xlim(x_left, x_right + x_right * 0.15)

        save_path = store_path / f"2_0_{c_class}_avg_shap_values"
        fig.savefig(f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_avg_shap_values_distributions(
    results: dict[str, Any], store_path: str | Path
) -> None:
    """Plot average SHAP values distributions (global effects distribution).

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)

    def _calculate_mean_abs_shap(
        folds_data: list, target_class: str
    ) -> tuple[dict, dict]:
        all_keys = {
            key
            for fold in folds_data
            for item in fold.get(target_class, [])
            for key in item.dict_values.keys()
        }
        fold_means = defaultdict(list)
        for fold in folds_data:
            class_fold = fold.get(target_class, [])
            for key in all_keys:
                abs_vals = [np.abs(item.dict_values.get(key, 0)) for item in class_fold]
                if abs_vals:
                    fold_means[key].append(np.nanmean(abs_vals))
                else:
                    fold_means[key].append(0.0)
        return {key: np.nanmean(vals) for key, vals in fold_means.items()}, fold_means

    x_names = results["config"]["X_NAMES"]
    params = results["params"]
    format_key = lambda key: " x ".join(str(x_names[idx]) for idx in key)

    for c_class in params["classes"]:
        shap_mean_abs, shap_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii"], c_class
        )
        shap_sh_mean_abs, shap_sh_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii_sh"], c_class
        )

        shapii_base = shap_mean_abs.pop((), 0.0)
        shap_sh_mean_abs.pop((), None)
        shap_folds_dct.pop((), None)
        shap_sh_folds_dct.pop((), None)

        shapii_se_sort = pd.Series(
            {format_key(k): v for k, v in shap_mean_abs.items()}
        ).sort_values(ascending=False)
        shapii_folds_df = pd.DataFrame(
            {format_key(k): v for k, v in shap_folds_dct.items()}
        )
        shapii_sh_folds_df = pd.DataFrame(
            {format_key(k): v for k, v in shap_sh_folds_dct.items()}
        )

        i_srt = shapii_se_sort.index
        shap_values_df_sort = shapii_folds_df.reindex(i_srt, axis=1)
        shap_values_sh_df_sort = shapii_sh_folds_df.reindex(i_srt, axis=1)

        shap_values_df_sort["Data"] = "original"
        shap_values_sh_df_sort["Data"] = "shuffled"

        value_name = "mean(|SHAP values|)"
        shap_values_df_sort_melt = shap_values_df_sort.melt(
            id_vars=["Data"], var_name="predictors", value_name=value_name
        )
        shap_values_sh_df_sort_melt = shap_values_sh_df_sort.melt(
            id_vars=["Data"], var_name="predictors", value_name=value_name
        )
        shap_values_df_sort_melt_all = pd.concat(
            [shap_values_df_sort_melt, shap_values_sh_df_sort_melt], axis=0
        )

        x_names_max_len = (
            max([len(i) for i in shapii_folds_df.columns.to_list()])
            if not shapii_folds_df.empty
            else 10
        )
        x_names_count = len(shapii_folds_df.columns.to_list())
        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
            f"mean(|SHAP values|): average effect on {params['y_name']}\n"
            f"mean(|SHAP values|): average change from expected value of {np.round(shapii_base, decimals=2)}"
        )
        if params["target_type"] in ["binary", "multiclass"]:
            title_str += f" (logit)\n class: {c_class}"
        title_lines_count = title_str.count("\n") + 1

        fig, ax = plt.subplots(
            figsize=(
                x_names_max_len * 0.1 + 8,
                x_names_count * 0.4 + title_lines_count * 0.4 + 0.5,
            ),
        )
        mypal = {"original": "#777777", "shuffled": "#eeeeee"}
        sns.violinplot(
            x=value_name,
            y="predictors",
            hue="Data",
            data=shap_values_df_sort_melt_all,
            bw_method="scott",
            bw_adjust=0.5,
            cut=2,
            density_norm="width",
            gridsize=100,
            width=0.8,
            inner="box",
            orient="h",
            linewidth=0.5,
            saturation=1,
            ax=ax,
            palette=mypal,
        )

        ax.set_xlabel("mean(|SHAP values|)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_ylabel("", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        ax.legend(loc="lower right")
        ax.set_title(title_str, fontsize=10)

        save_path = store_path / f"2_1_{c_class}_avg_shap_values_distributions"
        fig.savefig(f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_single_shap_values(results: dict[str, Any], store_path: str | Path) -> None:
    """Plot single SHAP values (beeswarm plot).

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)

    def _get_single_shap_values(folds_data: list, target_class: str) -> list:
        single_shap_values = []
        for fold in folds_data:
            single_shap_values.extend(fold.get(target_class, []))
        return single_shap_values

    x_names = results["config"]["X_NAMES"]
    params = results["params"]

    for c_class in params["classes"]:
        all_keys = {
            key
            for fold in results["total_shapii"]
            for item in fold.get(c_class, [])
            for key in item.dict_values.keys()
        }

        single_shap_values = _get_single_shap_values(results["total_shapii"], c_class)

        normalized_single_shap_values = []
        for iv in single_shap_values:
            new_dict_values = {key: iv.dict_values.get(key, 0.0) for key in all_keys}
            normalized_iv = shapiq.InteractionValues(
                values=new_dict_values,
                index=iv.index,
                max_order=iv.max_order,
                min_order=iv.min_order,
                n_players=iv.n_players,
                baseline_value=iv.baseline_value,
                estimation_budget=iv.estimation_budget,
                estimated=iv.estimated,
            )
            normalized_single_shap_values.append(normalized_iv)

        shapii_base = (
            np.mean([k.baseline_value for k in single_shap_values])
            if single_shap_values
            else 0.0
        )

        ax = shapiq.plot.beeswarm_plot(
            normalized_single_shap_values,
            results["total_x_tst_sample_shapiq"],
            max_display=len(all_keys),
            feature_names=x_names,
            abbreviate=False,
            alpha=0.5,
            row_height=0.2,
            ax=None,
            rng_seed=None,
            show=False,
        )

        fig = plt.gcf()
        ax.set_xlabel(ax.get_xlabel(), fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_ylabel(ax.get_ylabel(), fontsize=10)

        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
            f"SHAP values: single effects on {params['y_name']}\n"
            f"SHAP values: change from expected value of {np.round(shapii_base, decimals=2)}"
        )
        if params["target_type"] in ["binary", "multiclass"]:
            title_str += f" (logit)\n class: {c_class}"
        ax.set_title(title_str, fontsize=10)

        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=10)
        cb_ax.set_ylabel("Predictor value", fontsize=10)
        cb_ax.set_box_aspect(aspect=len(all_keys) * 7)

        save_path = store_path / f"2_2_{c_class}_shap_values"
        fig.savefig(f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_single_shap_values_dependences(
    results: dict[str, Any], store_path: str | Path
) -> None:
    """Plot single SHAP values dependence plots.

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the machine learning analyses.
    store_path : str or pathlib.Path
        Directory in which the figure is saved.

    Returns
    -------
    None
    """
    store_path = Path(store_path)

    def _get_single_shap_values(folds_data: list, target_class: str) -> list:
        single_shap_values = []
        for fold in folds_data:
            single_shap_values.extend(fold.get(target_class, []))
        return single_shap_values

    x_names = results["config"]["X_NAMES"]
    params = results["params"]
    format_key = lambda key: " x ".join(str(x_names[idx]) for idx in key)

    for c_class in params["classes"]:
        all_keys = {
            key
            for fold in results["total_shapii"]
            for item in fold.get(c_class, [])
            for key in item.dict_values.keys()
        }
        all_keys.discard(())

        single_shap_values = _get_single_shap_values(results["total_shapii"], c_class)

        normalized_single_shap_values = []
        for iv in single_shap_values:
            new_dict_values = {key: iv.dict_values.get(key, 0.0) for key in all_keys}
            normalized_iv = shapiq.InteractionValues(
                values=new_dict_values,
                index=iv.index,
                max_order=iv.max_order,
                min_order=iv.min_order,
                n_players=iv.n_players,
                baseline_value=iv.baseline_value,
                estimation_budget=iv.estimation_budget,
                estimated=iv.estimated,
            )
            normalized_single_shap_values.append(normalized_iv)

        shapii_base = (
            np.mean([k.baseline_value for k in single_shap_values])
            if single_shap_values
            else 0.0
        )

        for key in all_keys:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax = shapiq.plot.scatter_plot(
                normalized_single_shap_values,
                results["total_x_tst_sample_shapiq"],
                interaction=key,
                x_feature=key[0],
                color=key[1] if len(key) > 1 else None,
                feature_names=x_names,
                abbreviate=False,
                alpha=0.5,
                dot_size=16,
                jitter=0.01,
                hist=True,
                ax=ax,
                show=False,
            )

            ax.set_xlabel(ax.get_xlabel(), fontsize=10)
            ax.tick_params(axis="both", labelsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)

            title_str = (
                f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
                f"SHAP values: single effects on {params['y_name']}\n"
                f"SHAP values: change from expected value of {np.round(shapii_base, decimals=2)}"
            )
            if params["target_type"] in ["binary", "multiclass"]:
                title_str += f" (logit)\n class: {c_class}"
            ax.set_title(title_str, fontsize=10)

            key_str = format_key(key)
            save_path = (
                store_path / f"3_{c_class}_{key}_shap_values_dependency_{key_str}"
            )
            fig.savefig(
                f"{str(save_path)[:150]}.png", dpi=FIGURE_DPI, bbox_inches="tight"
            )
            plt.show()
            plt.close(fig)


def main() -> None:
    """Coordinate plotting and visualization of machine learning results.

    Returns
    -------
    None
    """
    setup_logging()
    logging.info("Plotting started.")
    document_requirements()

    if not RESULTS_DIR.exists():
        logging.warning("Results directory %s does not exist.", RESULTS_DIR)
        return

    targets_paths = [f.name for f in RESULTS_DIR.iterdir() if f.is_dir()]

    for target_path in targets_paths:
        logging.info("Target found: %s", target_path)

    for target_path in targets_paths:
        target_dir = RESULTS_DIR / target_path
        estimators_paths = [f.name for f in target_dir.iterdir() if f.is_dir()]

        for estimator_path in estimators_paths:
            logging.info("Estimator found: %s", estimator_path)

        for estimator_path in estimators_paths:
            estimator_dir = target_dir / estimator_path
            results_files_names = [
                f.name
                for f in estimator_dir.iterdir()
                if f.is_file() and f.suffix == ".pkl"
            ]

            results = {}
            for result_file_name in results_files_names:
                logging.info("Result found: %s", result_file_name)
                file_path = estimator_dir / result_file_name
                with file_path.open("rb") as filehandle:
                    key = result_file_name.split(".")[0]
                    results[key] = pkl.load(filehandle)

            if not results:
                continue

            current_plot_dir = PLOTS_DIR / target_path / estimator_path
            current_plot_dir.mkdir(parents=True, exist_ok=True)

            if results["params"]["pipe_name"] in ["linear", "gb"]:
                logging.info("Plotting optimized parameter distribution.")
                plot_parameter_distributions(results, current_plot_dir)

            target_type = results["params"]["target_type"]
            if target_type == "continuous":
                logging.info("Plotting regression model fit.")
                plot_regression_scatter(results, current_plot_dir)
                plot_regression_violin(results, current_plot_dir)
            elif target_type in ["binary", "multiclass"]:
                logging.info("Plotting classification model fit.")
                plot_classification_confusion(results, current_plot_dir)
                plot_classification_violin(results, current_plot_dir)
            else:
                raise ValueError(
                    f"Target type {target_type!r} was not in continuous/binary/multiclass."
                )

            logging.info("Plotting average SHAP values.")
            plot_avg_shap_values(results, current_plot_dir)

            logging.info("Plotting average SHAP values distribution.")
            plot_avg_shap_values_distributions(results, current_plot_dir)

            logging.info("Plotting single SHAP values.")
            plot_single_shap_values(results, current_plot_dir)

            logging.info("Plotting single SHAP values dependencies.")
            plot_single_shap_values_dependences(results, current_plot_dir)

    logging.info("Plotting finished.")


if __name__ == "__main__":
    main()
