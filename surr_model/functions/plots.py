
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from beartype import beartype
from scipy.stats import pearsonr
import sklearn.metrics as metrics 
import pandas as pd

@beartype
def linear_corr_plot(value_x: list, value_y: list, x_label: str, y_label: str,title:str) -> None:
    """
    both dictionary needs to be sorted in the same way
    """
    y_values = np.array(list(value_y))
    x_values = np.array(list(value_x))

    # Compute correlation
    r, p_value = pearsonr(y_values, x_values)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("talk")

    sns.regplot(
        x=x_values,
        y=y_values,
        ax=ax,
        scatter_kws={"s": 200, "alpha": 0.8},
        line_kws={"color": "red", "lw": 2},
        color="skyblue",
    )

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title("Correlation between Energy and Color variable")

    # Annotate correlation coefficient on the plot
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np = {p_value:.2e}\nMAE = {metrics.mean_absolute_error(value_y, value_x):.3f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    fig.suptitle(title)
    plt.show()
    
def rev_min_max_func(scaled_val):
    raw_train_ppi = pd.read_csv('/home/kunzj/BindCraft_uva_internship/data/ppi_affinity_dataset/raw/training.csv')
    max_val = np.max(raw_train_ppi['delta_G'])
    min_val = np.min(raw_train_ppi['delta_G'])
    #og_val = (scaled_val*(max_val - min_val)) + min_val

    og_val=  ((scaled_val + 1) * (max_val - min_val) / 2) + min_val
    return og_val


def linear_corr_plot_true_val(value_x: list, value_y: list, x_label: str, y_label: str,title:str) -> None:
    """
    both dictionary needs to be sorted in the same way
    """
    y_values = [rev_min_max_func(x) for x in list(value_y)]
    x_values = [rev_min_max_func(x) for x in list(value_x)]

    # Compute correlation
    r, p_value = pearsonr(y_values, x_values)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_context("talk")

    sns.regplot(
        x=x_values,
        y=y_values,
        ax=ax,
        scatter_kws={"s": 200, "alpha": 0.8},
        line_kws={"color": "red", "lw": 2},
        color="skyblue",
    )

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title("Correlation between Energy and Color variable")

    # Annotate correlation coefficient on the plot
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np = {p_value:.2e}\nMAE = {metrics.mean_absolute_error(y_values, x_values)/ 4.184:.3f} KCal/mol",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    fig.suptitle(title)
    plt.show()