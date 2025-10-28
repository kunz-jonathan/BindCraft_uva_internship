import math
import os

import equinox as eqx
import esm  # pip install fair-esm==2.0.0
import esm2quinox
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.random as jrandom
import numpy as np
import optax  # pip install optax
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, random_split
from colabdesign.af.alphafold.common import residue_constants
from .surr_model import AFF_PREDICTOR, init_surr_model, predict


def add_seq_loss(self, loss_weight: float) -> None:
    """
    wrapper around loss function that registers seq_loss in af_model

    Args:
        self (alf_model)
        loss_weight (flaot)

    Returns:
        None

    """
    #model, params_restored, rng = init_surr_model()
    model_key = jrandom.PRNGKey(0)
    torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    model_esm2 = esm2quinox.from_torch(torch_model)
    model_aff = AFF_PREDICTOR(model=model_esm2,key=model_key)


    aa_to_esm_array = jnp.array([ 5, 10, 17, 13, 23, 16,  9,  6, 21, 12,  4, 15, 20, 18, 14,  8, 11, 22, 19,  7])
    
    # --- JAX-compatible function ---
    def seq_to_esm_numbers(seq):
        seq = jnp.array(seq)
        seq_letters_idx = jnp.clip(seq, 0, 19)  # clip is based on colab dic
        seq_esm_numbers = aa_to_esm_array[seq_letters_idx]  # array indexing
        return jnp.array(seq_esm_numbers)

    def loss_fn( aux: dict)->float:
        """
        seq.-loss function
        
        Args:
            aux (dict): af2 dictionary containing all relevant information
            
        Returns:
            seq_loss (float): loss value
        
        """

        seq_esm = seq_to_esm_numbers(aux['seq']['pseudo'].argmax(-1))
        seq_loss = predict(model_aff,seq_esm).squeeze()
        return {"seq_loss": seq_loss}
        

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["seq_loss"] = loss_weight
