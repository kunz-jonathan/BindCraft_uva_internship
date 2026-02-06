import torch.nn.functional as F
import torch
from peft import LoraConfig, LoKrConfig,TaskType, get_peft_model
import equinox as eqx
import esm2quinox
import jax.random as jr
import jax
import torch.nn as nn
import jax.numpy as jnp
import optax
import esm
from transformers import AutoModel

class jax_predictor(eqx.Module):
    esm2_prot: esm2quinox.ESM2
    esm2_pept: esm2quinox.ESM2
    pooler_layer_prot: eqx.nn.Linear
    pooler_layer_pept: eqx.nn.Linear
    prot_droput: eqx.nn.Dropout
    pept_droput: eqx.nn.Dropout
    prot_projection: eqx.nn.Linear
    pept_projection: eqx.nn.Linear
    

    def __init__(self, model_prot,model_pept, key):
        key1, key2,key3,key4= jr.split(key, 4)

        self.esm2_prot = model_prot  # (num_layers=3, embed_size=32, num_heads=2, token_dropout=False, key=key)
        self.esm2_pept = model_pept
        # output size is 480
        self.pooler_layer_prot = eqx.nn.Linear(in_features= 640,out_features=640,key=key3)
        self.prot_droput = eqx.nn.Dropout(p=0.2)
        self.prot_projection = eqx.nn.Linear(in_features=640,out_features=320,key=key1)
        
        self.pooler_layer_pept = eqx.nn.Linear(in_features= 640,out_features=640,key=key4)
        self.pept_droput = eqx.nn.Dropout(p=0.2)
        self.pept_projection = eqx.nn.Linear(in_features=640,out_features=320,key=key2)
        

    def __call__(self, tokens_prot, tokens_pept, key):
        ### PROTEIN ###
        emb_prot = self.esm2_prot(tokens_prot).hidden # ([batch], seq_length, 320)
        if emb_prot.ndim == 3:
            # [B, L, H]
            emb_prot = emb_prot[:, 0]
        else:
            # [L, H]
            emb_prot = emb_prot[0]        
        x_prot = self.pooler_layer_prot(emb_prot)
        x_prot = jax.nn.tanh(x_prot)

        x_prot = self.prot_droput(x_prot,key=key)
        
        x_prot = self.prot_projection(x_prot)
        

        ### PEPTIDE ###
        emb_pept = self.esm2_pept(tokens_pept).hidden  # ([batch], seq_length, 320)
        if emb_pept.ndim == 3:
            # [B, L, H]
            emb_pept = emb_pept[:, 0]
        else:
            # [L, H]
            emb_pept = emb_pept[0]
        x_pept = self.pooler_layer_pept(emb_pept)
        x_pept = jax.nn.tanh(x_pept)
        x_pept = self.pept_droput(x_pept,key=key)
        x_pept = self.pept_projection(x_pept)


        ### PREDICTION-HEAD ###
        pred_aff = optax.cosine_similarity(x_prot,x_pept,axis=0,epsilon=1e-8) # to match the torch implementation
        

        return pred_aff
    