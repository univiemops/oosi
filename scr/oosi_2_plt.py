# -*- coding: utf-8 -*-
"""Out-of-sample inference 2 - Plotting
v001

@author: david.steyrl@univie.ac.at
"""

import importlib.metadata as im
import logging
import os
import pickle as pkl
import platform
from collections import defaultdict

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


def setup_logging() -> None:
    """Configure simultaneous logging to both a project file and the terminal."""
    # Basic logging
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/oosi_2_plt.log",
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
    logging.info(f"Logs save at: {log_dir}/oosi_2_plt.log")


def document_requirements() -> None:
    """Document environmental dependencies, logging system configuration and Python specs."""
    # Get distributions
    lines = [f"{dist.name}=={dist.version}" for dist in im.distributions()]

    # Write requirements
    with open("../logs/requirements_oosi_2_plt.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    logging.info(f"OS: {platform.system()} {platform.release()}")
    logging.info(f"Python Version: {platform.python_version()}")
    logging.info("Dependencies save at: ../logs/requirements_oosi_2_plt.txt")


def plot_parameter_distributions(results: dict, store_path: str) -> None:
    """Print model parameter distributions in histogram.

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """
    optimized_params = {
        key: [d[key] for d in results["total_best_params"]]
        for key in results["total_best_params"][0]
    }

    # Iterate over optimized_params
    for idx, (name, data) in enumerate(optimized_params.items()):
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(
            data=data,
            bins=30,
            kde=True,
            color="#777777",
            log_scale=(
                True
                if (
                    name.endswith("path_smooth")
                    or name.endswith("alpha")
                    or name.endswith("C")
                )
                else False
            ),
            ax=ax,
        )
        # Remove top, right and left frame elements
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add x label
        ax.set_xlabel(name.split("__")[-1])
        # Add y label
        ax.set_ylabel("Count")
        # Set x range
        if name.endswith("colsample_bytree"):
            ax.set_xlim([-0.05, 1.05])
        elif name.endswith("extra_trees"):
            ax.set_xlim([-0.05, 1.05])
        elif name.endswith("path_smooth"):
            ax.set_xlim([0.1, 100])
        elif name.endswith("alpha"):
            ax.set_xlim([0.01, 100])
        elif name.endswith("C"):
            ax.set_xlim([0.01, 100])
        elif name.endswith("l1_ratio"):
            ax.set_xlim([-0.05, 1.05])
        # Make title string
        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({results['params']['pipe_name']} estimator predicting {results['params']['y_name']})\n"
            f"Parameter distribution of predicting {results['params']['y_name']}"
        )
        # Set title
        ax.set_title(title_str, fontsize=10)

        # Make save path
        save_path = f"{store_path}/0_{idx}_opti_param_{name.split('__')[-1]}"[:150]
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        # Show figure
        plt.show()


def corrected_std(differences: np.ndarray, n_tst_over_n_trn: float = 0.25) -> float:
    """Corrects standard deviation using Nadeau and Bengio's approach.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    # Corrected variance
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_tst_over_n_trn)
    # Corrected standard deviation
    corrected_std = np.sqrt(corrected_var)
    # Return corrected standard deviation
    return corrected_std


def corrected_ttest(differences: np.ndarray, n_tst_over_n_trn: float = 0.25) -> float:
    """Computes right-tailed paired t-test with corrected variance.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    # Get mean of differences
    mean = np.nanmean(differences)
    # Get corrected standard deviation, make sure std is not exactly zero
    std = max(1e-6, corrected_std(differences, n_tst_over_n_trn))
    # Compute t statistics
    t_stat = mean / std
    # Compute p value for one-tailed t-test
    p_val = t.sf(t_stat, df=len(differences) - 1)
    # Return t statistics and p value
    return t_stat, p_val


def plot_regression_scatter(results: dict, store_path: str) -> None:
    """Model fit in a scatter plot (regression).

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """
    # Prepare results
    # True values per fold
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    # Predicted values
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    # True values
    true_values = np.concatenate(true_values_per_fold)
    # Predicted values
    pred_values = np.concatenate(pred_values_per_fold)
    # True values shuffle
    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    # Predicted values shuffle
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]
    # Compute MAE
    mae = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Extract MAE shuffle
    mae_sh = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Extract R²
    r2 = [r2_score(i, j) for i, j in zip(true_values_per_fold, pred_values_per_fold)]
    # Extract R² shuffle
    r2_sh = [
        r2_score(i, j) for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]

    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    ax.scatter(
        pred_values,
        true_values,
        zorder=2,
        alpha=0.1,
        color="#444444",
    )
    # Add optimal fit line
    ax.plot(
        [-10000, 10000],
        [-10000, 10000],
        color="#999999",
        zorder=3,
        linewidth=2,
        alpha=0.3,
    )
    # Fix aspect
    ax.set_aspect(1)
    # Remove top, right and left frame elements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Remove ticks
    ax.tick_params(
        axis="both",
        which="major",
        reset=True,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )
    # Add grid
    ax.grid(visible=True, which="major", axis="both")
    # Modify grid
    ax.tick_params(grid_linestyle=":", grid_alpha=0.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(
        min(true_values) - true_values_range / 20,
        max(true_values) + true_values_range / 20,
    )
    # Set y-axis limits
    ax.set_ylim(
        min(true_values) - true_values_range / 20,
        max(true_values) + true_values_range / 20,
    )
    title_str = f"{results['config']['ANALYSIS_NAME']} ({results['params']['pipe_name']} estimator predicting {results['params']['y_name']})\n"
    # Set title
    ax.set_title(title_str, fontsize=10)
    # Set xlabel
    ax.set_xlabel(f"Predicted {results['params']['y_name']}", fontsize=10)
    # Set x ticks size
    plt.xticks(fontsize=10)
    # Set ylabel
    ax.set_ylabel(f"True {results['params']['y_name']}", fontsize=10)
    # Set y ticks size
    plt.yticks(fontsize=10)

    # Add MAE
    # Calculate p-value between MAE and shuffle MAE
    _, pval_mae = corrected_ttest(np.array(mae_sh) - np.array(mae))
    # Add original outcome MAE results to plot
    ax.text(
        0.3,
        0.09,
        f"Original data: MAE mean{r'$\pm$'}std {np.nanmean(mae):.2f}{r'$\pm$'}{np.std(mae):.2f} | med {np.nanmedian(mae):.2f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    # Add shuffled outcome MAE results to plot
    ax.text(
        0.3,
        0.055,
        f"Shuffled data: MAE mean{r'$\pm$'}std {np.nanmean(mae_sh):.2f}{r'$\pm$'}{np.std(mae_sh):.2f} | med {np.nanmedian(mae_sh):.2f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    # If pval_mae <= 0.001
    if pval_mae <= 0.001:
        # Make pval string
        pval_string = "p\u22640.001"
    else:
        # Make pval string
        pval_string = f"p={pval_mae:.3f}"
    # Add p value to the plot
    ax.text(
        0.3,
        0.02,
        f"Original vs. shuffled: {pval_string}",
        transform=ax.transAxes,
        fontsize=10,
    )

    # Add R²
    # Calculate p-value between R² and shuffle R²
    _, pval_r2 = corrected_ttest(np.array(r2) - np.array(r2_sh))
    # Add original outcome R² results to plot
    ax.text(
        0.02,
        0.96,
        f"Original data: R² mean{r'$\pm$'}std {np.nanmean(r2):.3f}{r'$\pm$'}{np.std(r2):.3f} | med {np.nanmedian(r2):.3f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    # Add shuffled outcome R² results to plot
    ax.text(
        0.02,
        0.925,
        f"Shuffled data: R² mean{r'$\pm$'}std {np.nanmean(r2_sh):.3f}{r'$\pm$'}{np.std(r2_sh):.3f} | med {np.nanmedian(r2_sh):.3f}",
        transform=ax.transAxes,
        fontsize=10,
    )
    # If pval_r2 <= 0.001
    if pval_r2 <= 0.001:
        # Make pval string
        pval_string = "p\u22640.001"
    else:
        # Make pval string
        pval_string = "p={:.3f}".format(pval_r2)
    # Add p value to the plot
    ax.text(
        0.02,
        0.89,
        f"Original vs. shuffled: {pval_string}",
        transform=ax.transAxes,
        fontsize=10,
    )

    # Save figure
    # Make save path
    save_path = f"{store_path}/1_0_predictions"[:150]
    # Save figure
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # Show figure
    plt.show()


def plot_regression_violin(results: dict, store_path: str) -> None:
    """Model fit in a violin plot (regression).

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """
    # Prepare results
    # True values per fold
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    # Predicted values
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    # True values shuffle
    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    # Predicted values shuffle
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]
    # Compute MAE
    mae = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Extract MAE shuffle
    mae_sh = [
        mean_absolute_error(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Extract R²
    r2 = [r2_score(i, j) for i, j in zip(true_values_per_fold, pred_values_per_fold)]
    # Extract R² shuffle
    r2_sh = [
        r2_score(i, j) for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae)),
            "R²": pd.Series(np.array(r2)),
            "Data": pd.Series(["original" for _ in mae]),
            "Dummy": pd.Series(np.ones(np.array(mae).shape).flatten()),
        }
    )
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {
            "Mean Absolute Error": pd.Series(np.array(mae_sh)),
            "R²": pd.Series(np.array(r2_sh)),
            "Data": pd.Series(["shuffled" for _ in mae_sh]),
            "Dummy": pd.Series(np.ones(np.array(mae_sh).shape).flatten()),
        }
    )
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ["Mean Absolute Error", "R²"]

    # --- Make plot ---
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics) * 1 + 1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {"original": "#777777", "shuffled": "#eeeeee"}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
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
            ax=ax[i],
            palette=mypal,
        )
        # Remove top, right and left frame elements
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis="both",
            which="major",
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel("", fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(
            axis="y",
            color="#bbbbbb",
            linestyle="dotted",
            alpha=0.3,
        )
    # Make title string
    title_str = f"{results['config']['ANALYSIS_NAME']} ({results['params']['pipe_name']} estimator predicting {results['params']['y_name']})\n"
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # --- Save figure ---
    # Make save path
    save_path = f"{store_path}/1_1_predictions_distribution"[:150]
    # Save figure
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # Show plot
    plt.show()


def plot_classification_confusion(results: dict, store_path: str) -> None:
    """Model fit as confusion matrix plot (classification).

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """
    # Prepare results
    # True values per fold
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    # Predicted values
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    # True values
    true_values = np.concatenate(true_values_per_fold)
    # Predicted values
    pred_values = np.concatenate(pred_values_per_fold)
    # True values shuffle
    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    # Predicted values shuffle
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]
    # Accuracy
    acc = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Schuffle accuracy
    acc_sh = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Get classes
    class_labels = np.unique(true_values)

    # Get count confusion matrix
    # Loop over single results
    for true, pred in zip(true_values_per_fold, pred_values_per_fold):
        if "con_mat_count" not in locals():
            # Compute confusion matrix
            con_mat_count = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=None,
                normalize=None,
            )
        else:
            # Add confusion matrix
            con_mat_count = np.add(
                con_mat_count,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=None,
                    normalize=None,
                ),
            )

    # Get normalized confusion matrix
    # Loop over single results
    for true, pred in zip(true_values_per_fold, pred_values_per_fold):
        if "con_mat" not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=None,
                normalize="true",
            )
        else:
            # Add confusion matrix
            con_mat = np.add(
                con_mat,
                confusion_matrix(
                    true,
                    pred,
                    labels=class_labels,
                    sample_weight=None,
                    normalize="true",
                ),
            )
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values_per_fold)

    # Plot confusion matrix
    # Create figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(con_mat.shape[0] * 0.5 + 4, con_mat.shape[0] * 0.5 + 3.5),
    )
    # Use tight layout
    plt.tight_layout()
    # Plot count confusion matrix
    sns.heatmap(
        con_mat_count,
        vmin=None,
        vmax=None,
        cmap="Greys",
        center=None,
        robust=True,
        annot=True,
        fmt="",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        mask=None,
        ax=ax[0],
    )
    # Add title to plot
    ax[0].set_title("num. predictions", fontsize=10)
    # Add x label to plot
    ax[0].set_xlabel(f"Predicted {results['params']['y_name']}", fontsize=10)
    # Add y label to plot
    ax[0].set_ylabel(f"True {results['params']['y_name']}", fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    ax[0].tick_params(axis="y", labelsize=10, labelrotation=0)
    # Plot normalized confusion matrix
    sns.heatmap(
        con_mat_norm * 100,
        vmin=None,
        vmax=None,
        cmap="Greys",
        center=None,
        robust=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=1,
        linecolor="#999999",
        cbar=False,
        cbar_kws=None,
        square=True,
        xticklabels=[int(idx) for idx in class_labels],
        yticklabels=[int(idx) for idx in class_labels],
        mask=None,
        ax=ax[1],
    )
    # Add title to plot
    ax[1].set_title(f"norm. to True {results['params']['y_name']}", fontsize=10)
    # Add x label to plot
    ax[1].set_xlabel(f"Predicted {results['params']['y_name']}", fontsize=10)
    # Add y label to plot
    ax[1].set_ylabel(f"True {results['params']['y_name']}", fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    ax[1].tick_params(axis="y", labelsize=10, labelrotation=0)
    # Calculate p-value of accuracy and shuffle accuracy
    tstat_acc, pval_acc = corrected_ttest(np.array(acc) - np.array(acc_sh))
    # If pval_acc <= 0.001
    if pval_acc <= 0.001:
        # Make pval string
        pval_string = "p\u22640.001"
    else:
        # Make pval string
        pval_string = "p={:.3f}".format(pval_acc)
    # Make title string
    title_str = (
        f"{results['config']['ANALYSIS_NAME']} ({results['params']['pipe_name']} estimator predicting {results['params']['y_name']})\n"
        f"Original data acc: mean{r'$\pm$'}std {np.nanmean(acc):.3f}{r'$\pm$'}{np.std(acc):.3f} | med {np.nanmedian(acc):.3f}\n"
        f"Shuffled data acc: mean{r'$\pm$'}std {np.nanmean(acc_sh):.3f}{r'$\pm$'}{np.std(acc_sh):.3f} | med {np.nanmedian(acc_sh):.3f}\n"
        f"Original vs. shuffled: {pval_string}"
    )
    # Set title
    plt.suptitle(title_str, fontsize=10, y=0.95)

    # Save figure
    # Make save path
    save_path = f"{store_path}/1_0_predictions"[:150]
    # Save figure
    plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
    # Show figure
    plt.show()


def plot_classification_violin(results: dict, store_path: str) -> None:
    """Model fit in a violin plot (classification).

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """
    # Prepare results
    # True values per fold
    true_values_per_fold = [k["y_tst"] for k in results["total_scores"]]
    # Predicted values
    pred_values_per_fold = [k["y_pred"] for k in results["total_scores"]]
    # True values shuffle
    true_values_per_fold_sh = [k["y_tst"] for k in results["total_scores_sh"]]
    # Predicted values shuffle
    pred_values_per_fold_sh = [k["y_pred"] for k in results["total_scores_sh"]]
    # Accuracy
    acc = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold, pred_values_per_fold)
    ]
    # Schuffle accuracy
    acc_sh = [
        balanced_accuracy_score(i, j)
        for i, j in zip(true_values_per_fold_sh, pred_values_per_fold_sh)
    ]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc)),
            "Data": pd.Series(["original" for _ in acc]),
            "Dummy": pd.Series(np.ones(np.array(acc).shape).flatten()),
        }
    )
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {
            "Accuracy": pd.Series(np.array(acc_sh)),
            "Data": pd.Series(["shuffled" for _ in acc_sh]),
            "Dummy": pd.Series(np.ones(np.array(acc_sh).shape).flatten()),
        }
    )
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ["Accuracy"]

    # Make plot
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics) * 1 + 1))
    # Make color palette
    mypal = {"original": "#777777", "shuffled": "#eeeeee"}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
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
            ax=ax[i],
            palette=mypal,
        )
        # Remove top, right and left frame elements
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        # Remove ticks
        ax[i].tick_params(
            axis="both",
            which="major",
            reset=True,
            bottom=True,
            top=False,
            left=False,
            right=False,
            labelleft=False,
        )
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel("", fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
    # Make title string
    title_str = (
        f"{results['config']['ANALYSIS_NAME']} ({results['params']['pipe_name']} estimator predicting {results['params']['y_name']})\n"
        f"Predicting {results['params']['y_name']}"
    )
    # set title
    plt.title(title_str, fontsize=10)

    # Save figure
    # Make save path
    save_path = f"{store_path}/1_1_predictions_distribution"[:150]
    # Save figure
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # Show plot
    plt.show()


def plot_avg_shap_values(results: dict, store_path: str) -> None:
    """Plot average SHAP values (global effects).

    Parameters
    ----------
    results : dict
        Dictionary holding the results of the ml analyses.
    store_path : str
        Path to save the generated plots.
    """

    def _calculate_mean_abs_shap(folds_data: list, target_class: str) -> dict:
        """Helper to aggregate fold-level data into mean absolute SHAP values."""
        # Gather all unique keys across *all* items in *all* folds first
        all_keys = {
            key
            for fold in folds_data
            for item in fold.get(target_class, [])
            for key in item.dict_values.keys()
        }

        # Group values across folds by feature key
        fold_means = defaultdict(list)

        # Loop over the folds using the global set of keys
        for fold in folds_data:
            class_fold = fold.get(target_class, [])

            for key in all_keys:
                # Vectorized absolute mean across all items in this fold
                # Safe handling: if an item or key is missing, it defaults to 0
                abs_vals = [np.abs(item.dict_values.get(key, 0)) for item in class_fold]

                # Only append if there were actually items in this fold to prevent nanmean warnings
                if abs_vals:
                    fold_means[key].append(np.nanmean(abs_vals))
                else:
                    fold_means[key].append(0.0)

        # Calculate the overall mean across folds for each feature
        return {key: np.nanmean(vals) for key, vals in fold_means.items()}, fold_means

    x_names = results["config"]["X_NAMES"]
    params = results["params"]

    # Helper lambda to map a feature key tuple (e.g., (0, 3)) to its string representation
    format_key = lambda key: " x ".join(str(x_names[idx]) for idx in key)

    for c_class in params["classes"]:
        # Process True and Shuffled SHAP values using helper function
        shap_mean_abs, shap_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii"], c_class
        )
        shap_sh_mean_abs, shap_sh_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii_sh"], c_class
        )

        # Extract and remove the base value
        shapii_base = shap_mean_abs.pop((), 0.0)
        shap_sh_mean_abs.pop((), None)

        # Format into mapped pandas Series and sort
        shapii_se_sort = pd.Series(
            {format_key(k): v for k, v in shap_mean_abs.items()}
        ).sort_values(ascending=True)

        # Compute Statistical Significance (p-values)
        pval_dict = {}
        for key in shap_folds_dct.keys():
            if key == ():  # Skip base value
                continue
            # Vectorized element-wise difference between true and shuffled arrays
            diff = np.array(shap_folds_dct[key]) - np.array(shap_sh_folds_dct[key])
            _, c_pval = corrected_ttest(diff)
            pval_dict[format_key(key)] = c_pval

        pval_se = pd.Series(pval_dict)

        # Additional plot info
        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
            f"mean(|SHAP values|): average effect on {params['y_name']}\n"
            f"mean(|SHAP values|): average change from expected value of {np.round(shapii_base, decimals=2)}"
        )
        if params["target_type"] in ["binary", "multiclass"]:
            title_str += f" (logit)\n class: {c_class}"
        title_lines_count = title_str.count("\n") + 1
        x_names_max_len = max(len(str(i)) for i in shapii_se_sort.index)

        # Plot
        figsize = (
            x_names_max_len * 0.1 + 8,
            len(shapii_se_sort) * 0.4 + title_lines_count * 0.4 + 0.5,
        )
        fig, ax = plt.subplots(figsize=figsize)
        shapii_se_sort.plot(kind="barh", color="#777777", fontsize=10, ax=ax)

        # Formatting axes strings and frames
        ax.set_xlabel("mean(|SHAP values|)", fontsize=10)
        ax.set_title(title_str, fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)

        # Annotate Bars with Values and P-Values
        max_val = shapii_se_sort.max()
        for i, (c_pred, c_val) in enumerate(shapii_se_sort.items()):
            p_val = pval_se.get(c_pred, 1.0)
            pval_string = "p\u22640.001" if p_val <= 0.001 else f"p={p_val:.3f}"
            txt_str = f"{c_val:.2f} | {pval_string}"

            ax.text(
                c_val + max_val * 0.01,
                i,
                txt_str,
                color="k",
                va="center",
                fontsize=10,
            )

        # Adjust margins
        x_left, x_right = ax.get_xlim()
        ax.set_xlim(x_left, x_right + x_right * 0.15)

        # Save and Display
        save_path = os.path.join(store_path, f"2_0_{c_class}_avg_shap_values"[:150])
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_avg_shap_values_distributions(results: dict, store_path: str) -> None:
    """Plot average SHAP values distributions (global effects distribution).

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """

    def _calculate_mean_abs_shap(folds_data: list, target_class: str) -> dict:
        """Helper to aggregate fold-level data into mean absolute SHAP values."""
        # Gather all unique keys across *all* items in *all* folds first
        all_keys = {
            key
            for fold in folds_data
            for item in fold.get(target_class, [])
            for key in item.dict_values.keys()
        }

        # Group values across folds by feature key
        fold_means = defaultdict(list)

        # Loop over the folds using the global set of keys
        for fold in folds_data:
            class_fold = fold.get(target_class, [])

            for key in all_keys:
                # Vectorized absolute mean across all items in this fold
                # Safe handling: if an item or key is missing, it defaults to 0
                abs_vals = [np.abs(item.dict_values.get(key, 0)) for item in class_fold]

                # Only append if there were actually items in this fold to prevent nanmean warnings
                if abs_vals:
                    fold_means[key].append(np.nanmean(abs_vals))
                else:
                    fold_means[key].append(0.0)

        # Calculate the overall mean across folds for each feature
        return {key: np.nanmean(vals) for key, vals in fold_means.items()}, fold_means

    x_names = results["config"]["X_NAMES"]
    params = results["params"]

    # Helper lambda to map a feature key tuple (e.g., (0, 3)) to its string representation
    format_key = lambda key: " x ".join(str(x_names[idx]) for idx in key)

    # Loop over classes
    for c_class in params["classes"]:
        # Process True and Shuffled SHAP values using helper function
        shap_mean_abs, shap_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii"], c_class
        )
        shap_sh_mean_abs, shap_sh_folds_dct = _calculate_mean_abs_shap(
            results["total_shapii_sh"], c_class
        )

        # Extract and remove the base value
        shapii_base = shap_mean_abs.pop((), 0.0)
        shap_sh_mean_abs.pop((), None)
        shap_folds_dct.pop((), None)
        shap_sh_folds_dct.pop((), None)

        # Format into sorted mapped pandas Series and DataFrames
        shapii_se_sort = pd.Series(
            {format_key(k): v for k, v in shap_mean_abs.items()}
        ).sort_values(ascending=False)
        shapii_folds_df = pd.DataFrame(
            {format_key(k): v for k, v in shap_folds_dct.items()}
        )
        shapii_sh_folds_df = pd.DataFrame(
            {format_key(k): v for k, v in shap_sh_folds_dct.items()}
        )

        # Sorting index and reindex
        i_srt = shapii_se_sort.index
        shap_values_df_sort = shapii_folds_df.reindex(i_srt, axis=1)
        shap_values_sh_df_sort = shapii_sh_folds_df.reindex(i_srt, axis=1)

        # Add data origin
        shap_values_df_sort["Data"] = pd.DataFrame(
            ["original" for _ in range(shap_values_df_sort.shape[0])],
            columns=["Data"],
        )
        shap_values_sh_df_sort["Data"] = pd.DataFrame(
            ["shuffled" for _ in range(shap_values_sh_df_sort.shape[0])],
            columns=["Data"],
        )

        # Value name, melt dataframes and concatenate dataframes
        value_name = "mean(|SHAP values|)"
        shap_values_df_sort_melt = shap_values_df_sort.melt(
            id_vars=["Data"],
            var_name="predictors",
            value_name=value_name,
        )
        shap_values_sh_df_sort_melt = shap_values_sh_df_sort.melt(
            id_vars=["Data"],
            var_name="predictors",
            value_name=value_name,
        )
        shap_values_df_sort_melt_all = pd.concat(
            [shap_values_df_sort_melt, shap_values_sh_df_sort_melt],
            axis=0,
        )

        # Additional info
        x_names_max_len = max([len(i) for i in shapii_folds_df.columns.to_list()])
        x_names_count = len(shapii_folds_df.columns.to_list())
        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
            f"mean(|SHAP values|): average effect on {params['y_name']}\n"
            f"mean(|SHAP values|): average change from expected value of {np.round(shapii_base, decimals=2)}"
        )
        if params["target_type"] in ["binary", "multiclass"]:
            title_str += f" (logit)\n class: {c_class}"
        title_lines_count = title_str.count("\n") + 1

        # Plot
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

        # Formatting axes strings and frames
        _, ax = plt.gcf(), plt.gca()
        plt.xlabel("mean(|SHAP values|)", fontsize=10)
        plt.xticks(fontsize=10)
        plt.ylabel("", fontsize=10)
        plt.yticks(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#bbbbbb", linestyle="dotted", alpha=0.3)
        plt.legend(loc="lower right")
        ax.set_title(title_str, fontsize=10)

        # Save and show
        save_path = os.path.join(
            store_path, f"2_1_{c_class}_avg_shap_values_distributions"[:150]
        )
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_single_shap_values(results: dict, store_path: str) -> None:
    """Plot single SHAP values.

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """

    def _get_single_shap_values(folds_data: list, target_class: str) -> dict:
        """Helper to aggregate fold-level data into mean absolute SHAP values."""
        single_shap_values = []

        # Loop over the fold to get the single shap value data
        for fold in folds_data:
            single_shap_values.extend(fold.get(target_class, []))

        return single_shap_values

    x_names = results["config"]["X_NAMES"]
    params = results["params"]

    # Loop over classes
    for c_class in params["classes"]:
        # Gather all unique keys across *all* items in *all* folds first
        all_keys = {
            key
            for fold in results["total_shapii"]
            for item in fold.get(c_class, [])
            for key in item.dict_values.keys()
        }

        # Get current shap values
        single_shap_values = _get_single_shap_values(results["total_shapii"], c_class)

        # Get mean base value
        shapii_base = np.mean([k.baseline_value for k in single_shap_values])

        # Plot
        ax = shapiq.plot.beeswarm_plot(
            single_shap_values,
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

        # Adjust fontsizes of axes
        fig = plt.gcf()
        plt.xlabel(ax.get_xlabel(), fontsize=10)
        plt.xticks(fontsize=10)
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        plt.yticks(fontsize=10)

        # Add title
        title_str = (
            f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
            f"SHAP values: single effects on {params['y_name']}\n"
            f"SHAP values: change from expected value of {np.round(shapii_base, decimals=2)}"
        )
        if params["target_type"] in ["binary", "multiclass"]:
            title_str += f" (logit)\n class: {c_class}"
        plt.title(title_str, fontsize=10)

        # Color bar
        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=10)
        cb_ax.set_ylabel("Predictor value", fontsize=10)
        cb_ax.set_box_aspect(aspect=len(all_keys) * 7)

        # Save and Display
        save_path = f"{store_path}/2_2_{c_class}_shap_values"[:150]
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_single_shap_values_dependences(results: dict, store_path: str) -> None:
    """Plot single SHAP values dependences.

    Parameters
    ----------
    results : dictionary
        Dictionary holding the results of the ml analyses.
    store_path : string
        Path to the plots.
    """

    def _get_single_shap_values(folds_data: list, target_class: str) -> dict:
        """Helper to aggregate fold-level data into mean absolute SHAP values."""
        single_shap_values = []

        # Loop over the fold to get the single shap value data
        for fold in folds_data:
            single_shap_values.extend(fold.get(target_class, []))

        return single_shap_values

    x_names = results["config"]["X_NAMES"]
    params = results["params"]

    # Helper lambda to map a feature key tuple (e.g., (0, 3)) to its string representation
    format_key = lambda key: " x ".join(str(x_names[idx]) for idx in key)

    # Loop over classes
    for c_class in params["classes"]:
        # Gather all unique keys across *all* items in *all* folds first
        all_keys = {
            key
            for fold in results["total_shapii"]
            for item in fold.get(c_class, [])
            for key in item.dict_values.keys()
        }
        all_keys.remove(())

        # Get current shap values
        single_shap_values = _get_single_shap_values(results["total_shapii"], c_class)

        # Get mean base value
        shapii_base = np.mean([k.baseline_value for k in single_shap_values])

        # Loop over predictors
        for idx, key in enumerate(all_keys):
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot
            ax = shapiq.plot.scatter_plot(
                single_shap_values,
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

            # Adjust fontsizes of axes
            plt.xlabel(ax.get_xlabel(), fontsize=10)
            plt.xticks(fontsize=10)
            plt.ylabel(ax.get_ylabel(), fontsize=10)
            plt.yticks(fontsize=10)

            # Title
            title_str = (
                f"{results['config']['ANALYSIS_NAME']} ({params['pipe_name']} estimator predicting {params['y_name']})\n"
                f"SHAP values: single effects on {params['y_name']}\n"
                f"SHAP values: change from expected value of {np.round(shapii_base, decimals=2)}"
            )
            if params["target_type"] in ["binary", "multiclass"]:
                title_str += f" (logit)\n class: {c_class}"
            plt.title(title_str, fontsize=10)

            # Save and Display
            save_path = f"{store_path}/3_{c_class}_{key}_shap_values_dependency_{format_key(key)}"[
                :150
            ]
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
            plt.show()


def main() -> None:
    """Coordinate plotting."""
    setup_logging()
    logging.info("Plotting started.")
    document_requirements()
    targets_paths = [f.name for f in os.scandir(path="../results")]

    for target_path in targets_paths:
        logging.info(f"Target found: {target_path}")

    # Loop over targets paths
    for target_path in targets_paths:
        # Get estimators paths
        estimators_paths = [
            f.name for f in os.scandir(path=f"../results/{target_path}")
        ]

        for estimator_path in estimators_paths:
            logging.info(f"Estimator found: {estimator_path}")

        # Loop over estimators paths
        for estimator_path in estimators_paths:
            # Get results files names
            results_files_names = [
                f.name
                for f in os.scandir(path=f"../results/{target_path}/{estimator_path}")
            ]

            # Loop over results files names and load results
            results = {}
            for result_file_name in results_files_names:
                logging.info(f"Result found: {result_file_name}")

                # Load results
                with open(
                    f"../results/{target_path}/{estimator_path}/{result_file_name}",
                    "rb",
                ) as filehandle:
                    results[result_file_name.split(".")[0]] = pkl.load(filehandle)

            # Create directory
            os.makedirs(f"../plots/{target_path}/{estimator_path}", exist_ok=True)

            # Plot optimized parameter
            if results["params"]["pipe_name"] in ["linear", "gb"]:
                logging.info("Plot optimized parameter distribution.")

                plot_parameter_distributions(
                    results, f"../plots/{target_path}/{estimator_path}"
                )

            # Plot model fit
            if results["params"]["target_type"] == "continuous":
                logging.info("Plot model fit.")
                # Print model fit as scatter plot
                plot_regression_scatter(
                    results, f"../plots/{target_path}/{estimator_path}"
                )
                # Print model fit as violinplot of metrics
                plot_regression_violin(
                    results, f"../plots/{target_path}/{estimator_path}"
                )
            elif results["params"]["target_type"] in ["binary", "multiclass"]:
                logging.info("Plot model fit.")
                # Print model fit as confusion matrix
                plot_classification_confusion(
                    results, f"../plots/{target_path}/{estimator_path}"
                )
                # Print model fit as violinplot of metrics
                plot_classification_violin(
                    results, f"../plots/{target_path}/{estimator_path}"
                )
            else:
                raise ValueError(
                    f"Target index {results['params']['target_type']} was not in continuous/binary/multiclass."
                )

            # Plot average SHAP values
            logging.info("Plot average SHAP values.")
            plot_avg_shap_values(results, f"../plots/{target_path}/{estimator_path}")

            # Plot average SHAP values distribution
            logging.info("Plot average SHAP values distribution.")
            plot_avg_shap_values_distributions(
                results, f"../plots/{target_path}/{estimator_path}"
            )

            # Plot single SHAP values
            logging.info("Plot single SHAP values.")
            plot_single_shap_values(results, f"../plots/{target_path}/{estimator_path}")

            # Plot single SHAP values dependencies
            logging.info("Plot single SHAP values dependencies.")
            plot_single_shap_values_dependences(
                results, f"../plots/{target_path}/{estimator_path}"
            )

    logging.info("Plotting finished.")


if __name__ == "__main__":
    main()
