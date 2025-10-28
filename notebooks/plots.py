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