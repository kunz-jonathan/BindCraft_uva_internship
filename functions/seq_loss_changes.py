import math
import os
import sys

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
from surr_model.functions.model import AFF_PREDICTOR, stripped_PREDICTOR
import pickle
import jax.tree_util as jtu

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

#### 
#
# MAYBE CHANGE THE ESM2 MODEL SIZE TO SMALLER AND THEN IT MAY WORK ????
#
####
def add_seq_loss(self, loss_weight: float) -> None:
    """
    Registers seq_loss in af_model without tracing surrogate model parameters.
    Gradients do not flow through the surrogate.
    """

    # -------------------------------
    # Step 0: Keys & model
    model_key, call_key = jr.split(jr.PRNGKey(0), 2)
    torch_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    model_esm2 = esm2quinox.from_torch(torch_model)
    model_aff, model_state = eqx.nn.make_with_state(stripped_PREDICTOR)(
        model=model_esm2, key=model_key
    )

    # inference mode
    inference_model = eqx.nn.inference_mode(model_aff)
    inference_model = eqx.Partial(inference_model, state=model_state)

    # load pretrained weights
    best_model_aff = eqx.tree_deserialise_leaves(
        "/home/kunzj/BindCraft_uva_internship/surr_model/params/model_inference_second_training_set.eqx",
        inference_model,
    )

    # -------------------------------
    # Step 1: Partition model into params & static
    params, static = eqx.partition(best_model_aff, eqx.is_array)

    # -------------------------------
    # Step 2: target protein sequence (batch=1)
    x_prot = jnp.array(
        [
            0,
            16,
            14,
            10,
            6,
            6,
            6,
            14,
            11,
            8,
            8,
            9,
            16,
            12,
            20,
            15,
            11,
            6,
            5,
            4,
            4,
            4,
            16,
            6,
            18,
            12,
            16,
            13,
            10,
            5,
            6,
            10,
            20,
            6,
            6,
            9,
            5,
            14,
            9,
            4,
            5,
            4,
            13,
            14,
            7,
            14,
            16,
            13,
            5,
            8,
            11,
            15,
            15,
            4,
            8,
            9,
            23,
            4,
            15,
            10,
            12,
            6,
            13,
            9,
            4,
            13,
            8,
            17,
            20,
            9,
            4,
            16,
            10,
            20,
            12,
            5,
            5,
            7,
            13,
            11,
            13,
            8,
            14,
            10,
            9,
            7,
            18,
            18,
            10,
            7,
            5,
            5,
            13,
            20,
            18,
            8,
            13,
            6,
            17,
            18,
            17,
            22,
            6,
            10,
            7,
            7,
            5,
            4,
            18,
            19,
            18,
            5,
            8,
            15,
            4,
            7,
            4,
            15,
            5,
            4,
            23,
            11,
            15,
            7,
            14,
            9,
            4,
            12,
            10,
            11,
            12,
            20,
            6,
            22,
            11,
            4,
            13,
            18,
            4,
            10,
            9,
            10,
            4,
            4,
            6,
            22,
            12,
            16,
            13,
            16,
            6,
            6,
            22,
            13,
            6,
            4,
            4,
            8,
            19,
            18,
            6,
            2,
        ]
    )

    call_key = jr.split(call_key, x_prot.shape[0])

    # -------------------------------
    # Step 3: map AF2 indices to ESM2
    aa_to_esm_array = jnp.array(
        [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7]
    )

    def seq_to_esm_numbers(seq):
        seq_letters_idx = jnp.clip(seq, 0, 19)
        return aa_to_esm_array[seq_letters_idx]

    # -------------------------------
    # Step 4: JAX-compatible seq_loss function
    @eqx.filter_jit
    def loss_fn(aux: dict) -> dict:
        seq_esm = seq_to_esm_numbers(aux["seq"]["pseudo"].argmax(-1)).squeeze()

        # Recombine model for forward pass
        model = eqx.combine(params, static)

        # Forward pass through surrogate — stop gradients
        pred_y, _ = jax.lax.stop_gradient(model(x_prot, seq_esm, key=call_key))

        seq_loss = jnp.square(jax.nn.relu(pred_y)).squeeze()
        return {"seq_loss": seq_loss}

    # -------------------------------
    # Step 5: register callback
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["seq_loss"] = loss_weight


# def add_seq_loss(self, loss_weight: float) -> None:
#     """
#     wrapper around loss function that registers seq_loss in af_model

#     Args:
#         self (alf_model)
#         loss_weight (flaot)

#     Returns:
#         None

#     """
#     # generating keys
#     model_key, call_key = jr.split(jrandom.PRNGKey(0), 2)

#     # initializing models
#     torch_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
#     model_esm2 = esm2quinox.from_torch(torch_model)
#     model_aff, model_state = eqx.nn.make_with_state(stripped_PREDICTOR)(
#         model=model_esm2, key=model_key
#     )

#     # set to inference mode
#     inference_model = eqx.nn.inference_mode(model_aff)
#     inference_model = eqx.Partial(inference_model, state=model_state)

#     # load best inference pretrained model
#     best_model_aff = eqx.tree_deserialise_leaves('/home/kunzj/BindCraft_uva_internship/surr_model/params/model_inference_second_training_set.eqx',inference_model)

#     params, static = eqx.partition(best_model_aff, eqx.is_array)

#     # define target sequence
#     x_prot =jnp.array([ 0, 16, 14, 10,  6,  6,  6, 14, 11,  8,  8,  9, 16, 12, 20, 15,
#             11,  6,  5,  4,  4,  4, 16,  6, 18, 12, 16, 13, 10,  5,  6, 10,
#             20,  6,  6,  9,  5, 14,  9,  4,  5,  4, 13, 14,  7, 14, 16, 13,
#             5,  8, 11, 15, 15,  4,  8,  9, 23,  4, 15, 10, 12,  6, 13,  9,
#             4, 13,  8, 17, 20,  9,  4, 16, 10, 20, 12,  5,  5,  7, 13, 11,
#             13,  8, 14, 10,  9,  7, 18, 18, 10,  7,  5,  5, 13, 20, 18,  8,
#             13,  6, 17, 18, 17, 22,  6, 10,  7,  7,  5,  4, 18, 19, 18,  5,
#             8, 15,  4,  7,  4, 15,  5,  4, 23, 11, 15,  7, 14,  9,  4, 12,
#             10, 11, 12, 20,  6, 22, 11,  4, 13, 18,  4, 10,  9, 10,  4,  4,
#             6, 22, 12, 16, 13, 16,  6,  6, 22, 13,  6,  4,  4,  8, 19, 18,
#             6,  2])

#     call_key = jr.split(call_key, x_prot.shape[0])

#     # map af2 colabdesign aa-dic to esm2 equinox aa-dic
#     aa_to_esm_array = jnp.array(
#         [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7]
#     )

#     # --- JAX-compatible function ---
#     def seq_to_esm_numbers(seq):
#         seq = jnp.array(seq)
#         seq_letters_idx = jnp.clip(seq, 0, 19)  # clip is based on colab dic
#         seq_esm_numbers = aa_to_esm_array[seq_letters_idx]  # array indexing
#         return jnp.array(seq_esm_numbers)


#     @eqx.filter_jit
#     def loss_fn(aux: dict) -> float:
#         """
#         seq.-loss function

#         Args:
#             aux (dict): af2 dictionary containing all relevant information

#         Returns:
#             seq_loss (float): loss value

#         """

#         seq_esm = seq_to_esm_numbers(aux["seq"]["pseudo"].argmax(-1)).squeeze()

#         #pred_y, _ = jax.vmap(best_model_aff)(x_prot, seq_esm, key=call_key)
#         model = eqx.combine(params, static)
#         pred_y, _ =  jax.lax.stop_gradient(model(x_prot, seq_esm, state=None, key=call_key))

#         seq_loss = jnp.square(jax.nn.relu(pred_y)).squeeze()
#         return {"seq_loss": seq_loss}

#     self._callbacks["model"]["loss"].append(loss_fn)
#     self.opt["weights"]["seq_loss"] = loss_weight
