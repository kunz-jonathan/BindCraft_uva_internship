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
from surr_model.functions.model import AFF_PREDICTOR
import pickle

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
    model_aff, model_state = eqx.nn.make_with_state(AFF_PREDICTOR)(
        model=model_esm2, key=model_key
    )
    # load trained model
    best_model_aff = eqx.tree_deserialise_leaves('/home/jkunz/master_uni_hd/internship_amsterdam/BindCraft_uva_internship/surr_model/params/params_only_seq.pkl',model_aff)
    best_model_state = pickle.load(open('/home/jkunz/master_uni_hd/internship_amsterdam/BindCraft_uva_internship/surr_model/params/state_only_seq.pkl','rb'))

    # set to inference mode
    inference_model = eqx.nn.inference_mode(best_model_aff)
    inference_model = eqx.Partial(inference_model, state=best_model_state)
    
    # define target sequence
    target_str =['QPRGGGPTSSEQIMKTGALLLQGFIQDRAGRMGGEAPELALDPVPQDASTKKLSECLKRIGDELDSNMELQRMIAAVDTDSPREVFFRVAADMFSDGNFNWGRVVALFYFASKLVLKALCTKVPELIRTIMGWTLDFLRERLLGWIQDQGGWDGLLSYFG']
    x_prot = esm2quinox.tokenise(target_str)

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

    def loss_fn(aux: dict) -> float:
        """
        seq.-loss function

        Args:
            aux (dict): af2 dictionary containing all relevant information

        Returns:
            seq_loss (float): loss value

        """

        seq_esm = seq_to_esm_numbers(aux["seq"]["pseudo"].argmax(-1))
        pred_y, _ = jax.vmap(inference_model)(x_prot, seq_esm, key=call_key).squeeze()
        #  pred_y from -1,+1 scale so that -1 low loss and +1 high loss
        seq_loss = pred_y
        return {"seq_loss": seq_loss}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["seq_loss"] = loss_weight
