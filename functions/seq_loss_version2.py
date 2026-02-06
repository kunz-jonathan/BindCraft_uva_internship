import math
import os
import sys

import equinox as eqx
import esm  # pip install fcair-esm==2.0.0
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
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def add_seq_loss(self, loss_weight: float) -> None:
    """
    wrapper around loss function that registers seq_loss in af_model

    Args:
        self (alf_model)
        loss_weight (flaot)

    Returns:
        None

    """
    # generating keys
    model_key, call_key = jr.split(jrandom.PRNGKey(0), 2)

    # initializing models
    torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    model_esm2 = esm2quinox.from_torch(torch_model)
    model_aff, model_state = eqx.nn.make_with_state(stripped_PREDICTOR)(
        model=model_esm2, key=model_key
    )

    # set to inference mode
    inference_model = eqx.nn.inference_mode(model_aff)
    inference_model = eqx.Partial(inference_model, state=model_state)

    # load best inference pretrained model
    # best_model_aff = eqx.tree_deserialise_leaves('/home/kunzj/BindCraft_uva_internship/surr_model/params/model_inference_second_training_set.eqx',inference_model)



    # define target sequence
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

    # map af2 colabdesign aa-dic to esm2 equinox aa-dic
    aa_to_esm_array = jnp.array(
        [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7]
    )

    # --- JAX-compatible function ---
    def seq_to_esm_numbers(seq):
        seq = jnp.array(seq)
        seq_letters_idx = jnp.clip(seq, 0, 19)  # clip is based on colab dic
        seq_esm_numbers = aa_to_esm_array[seq_letters_idx]  # array indexing
        return jnp.array(seq_esm_numbers)
    
    @eqx.filter_jit
    def loss_fn(aux: dict) -> float:
        """
        seq.-loss function

        Args:
            aux (dict): af2 dictionary containing all relevant information

        Returns:
            seq_loss (float): loss value

        """
        seq_esm = seq_to_esm_numbers(aux["seq"]["pseudo"].argmax(-1))

        # batched version
        # pred_y, _ = jax.vmap(inference_model)(x_prot, seq_esm, key=call_key)

        pred_y, _ = jax.lax.stop_gradient(
            inference_model(x_prot, seq_esm.squeeze(), key=call_key)
        )
        seq_loss = jax.nn.relu(pred_y).squeeze()

        return {"seq_loss": seq_loss}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["seq_loss"] = loss_weight

    ###PYTORCH WAY###\
    # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    # model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    # # Example protein sequences (as strings of amino‑acid letters)
    # sequences = ["MAVLKQ", "MGHWTELS"]  # replace with your protein sequences

    # # Tokenize (handles padding if you pass multiple sequences)
    # inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)

    # # Move to device
    # inputs = {key: val for key, val in inputs.items()}
    # out = model(**inputs)['pooler_output']
    # seq_loss = np.mean(out.detach().numpy())
