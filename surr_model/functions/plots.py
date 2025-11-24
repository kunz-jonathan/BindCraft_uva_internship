from pathlib import Path
from typing import List

import equinox as eqx
import esm  # pip install fair-esm==2.0.0
import esm2quinox
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import optax  # pip install optax
import pandas as pd
import seaborn as sns
from beartype import beartype
from jaxtyping import Array, Int
from scipy.stats import zscore
from torch.utils.data import DataLoader, RandomSampler, random_split
from sklearn import preprocessing
from scipy.stats import pearsonr


@beartype
def linear_corr_plot(value_x: list, value_y: list, x_label: str, y_label: str) -> None:
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
        f"r = {r:.3f}\np = {p_value:.2e}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    plt.show()