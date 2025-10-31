import equinox as eqx
import esm2quinox
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable


####========NONSENSE========####
# class SelfAttention(eqx.Module):
#     query: eqx.nn.Linear
#     key: eqx.nn.Linear
#     value: eqx.nn.Linear
#     softmax: jax.nn.softmax

#     def __init__(self,hidden_dim, key):
#         key1, key2, key3 = jr.split(key, 3)
#         super(SelfAttention, self).__init__()
#         self.query = eqx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, key=key1)
#         self.key = eqx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, key=key2)
#         self.value = eqx.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, key=key3)
#         self.softmax = jax.nn.softmax(axis)


#     def __call__(self,x):
#         query = self.key(x)
#         key= self.key(x)
#         value = self.value(x)
#         scores = jnp.matmul(query, key.transpose()) / (x.size ** 0.5)
#         attention = self.softmax(scores) # softmax is across your batch dimension
#         output = jnp.matmul(attention,value)
#         return output
class Prediction_Head(eqx.Module):
    layers: list[eqx.nn.Linear | eqx.nn.Dropout | Callable]

    def __init__(self, key):
        super().__init__()
        key1, key2, key3, key4 = jr.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_features=128, out_features=64, key=key1),
            jax.nn.relu,
            eqx.nn.Dropout(p=0.1),
            eqx.nn.Linear(in_features=64, out_features=32, key=key2),
            jax.nn.hard_tanh,
            eqx.nn.Linear(in_features=32, out_features=1, key=key3),
        ]

    def __call__(self, x, key):
        
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x,key=  key)
            else:
                x = layer(x)
        return x


"""
class AFF_PREDICTOR_old(eqx.Module):
    esm2: esm2quinox.ESM2
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    conv3: eqx.nn.Conv1d
    batchnorm1: eqx.nn.BatchNorm
    batchnorm2: eqx.nn.BatchNorm
    batchnorm3: eqx.nn.BatchNorm
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    prediction_head: Prediction_Head
    # relu: jax.nn.relu
    #SelfAttention: eqx.Module

    def __init__(self, model, key):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)
        out_dim_cnn = ...
        self.esm2 = model  # (num_layers=3, embed_size=32, num_heads=2, token_dropout=False, key=key)
        # output size is 480
        

        
        self.conv1 = eqx.nn.Conv1d(
            in_channels=320,
            out_channels=256,
            kernel_size=5,
            padding="SAME",
            stride=1,
            key=key1,
        )
        self.conv2 = eqx.nn.Conv1d(
            in_channels=256,
            out_channels=128,
            kernel_size=5,
            padding="SAME",
            stride=1,
            key=key2,
        )
        self.conv3 = eqx.nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            padding="SAME",
            stride=1,
            key=key3,
        )
        self.batchnorm1 = eqx.nn.BatchNorm(256, axis_name="batch")
        self.batchnorm2 = eqx.nn.BatchNorm(128, axis_name="batch")
        self.batchnorm3 = eqx.nn.BatchNorm(64, axis_name="batch")

        
        self.prediction_head = Prediction_Head(key5,hidden_dim=out_dim_cnn)


    def __call__(self, tokens_prot,tokens_pept, state):
        
        # generate protein hidden dimension
        out_esm_prot = self.esm2(tokens_prot).hidden  # ([batch], seq_length, 320)
        out_esm_prot = jnp.transpose(out_esm_prot, (1, 0))  # out: ([batch], 320, seq_length)
        out_esm_prot = jnp.array(out_esm_prot)
        
        # generate peptide hidden dimension
        out_esm_pept = self.esm2(tokens_pept).hidden  # ([batch], seq_length, 320)
        out_esm_pept = jnp.transpose(out_esm_pept, (1, 0))  # out: ([batch], 320, seq_length)
        out_esm_pept = jnp.array(out_esm_pept)
        
        
        # simple cnn 1d stack with batchnorm for protein and peptide individually ~ can be extended to attention mechanism, not sure if necessar
        out_conv1 = self.conv1(out_esm_conc)
        out_batchn1, state1 = self.batchnorm1(out_conv1, state)
        out_conv2 = self.conv2(out_batchn1)
        out_batchn2, state2 = self.batchnorm2(out_conv2, state1)
        out_conv3 = self.conv3(out_batchn2)
        out_batchn3, state3 = self.batchnorm3(out_conv3, state2)
        
        # simple mean along embedding, however could be extended to attention_mean but not sure if necessary? 
        out_pool = jnp.mean(out_batchn3, axis=1, keepdims=True) #mean along the sequence so that out: ([batch],embedding,1)
        
        out_pool_ord = jnp.transpose(out_pool, (1, 0)) # out: ([batch], 1, embedding)
        out_pool_ord = out_pool_ord.squeeze() # out: ([batch], embedding)
        
        out_attention_head = self.prediction_head(out_pool_ord)
        
        return out_attention_head, state3
"""


class AFF_PREDICTOR(eqx.Module):
    esm2: esm2quinox.ESM2
    cnn_stack: list[eqx.nn.Conv1d | eqx.nn.Dropout | Callable]
    prediction_head: Prediction_Head

    def __init__(self, model, key):
        key1, key2, key3, key4, key5, key6 = jr.split(key, 6)

        self.esm2 = model  # (num_layers=3, embed_size=32, num_heads=2, token_dropout=False, key=key)
        # output size is 480

        self.cnn_stack = [
            eqx.nn.Conv1d(
                in_channels=320,
                out_channels=256,
                kernel_size=5,
                padding="SAME",
                stride=1,
                key=key1,
            ),
            eqx.nn.Dropout(p=0.1),
            eqx.nn.Conv1d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                padding="SAME",
                stride=1,
                key=key2,
            ),
            eqx.nn.Dropout(p=0.1),
            eqx.nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                padding="SAME",
                stride=1,
                key=key3,
            ),
        ]

        self.prediction_head = Prediction_Head(key5)

    def __call__(self, tokens_prot, tokens_pept, key):
        ### PROTEIN ###

        # generate protein hidden dimension
        emb_prot = self.esm2(tokens_prot).hidden  # ([batch], seq_length, 320)
        emb_prot = jnp.transpose(emb_prot, (1, 0))  # out: ([batch], 320, seq_length)
        x = jnp.array(emb_prot)
        print(x.shape)
        # protein cnn stack
        for layer in self.cnn_stack:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x,key= key)
            else:
                x = layer(x)

        # simple mean along embedding, however could be extended to attention_mean, not sure if necessary
        x_pooled = jnp.mean(
            x, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)

        x_pooled = jnp.transpose(x_pooled, (1, 0))  # out: ([batch], 1, embedding)
        x_protein = x_pooled.squeeze()  # out: ([batch], embedding)

        ### PEPTIDE ###

        # generate peptide hidden dimension
        emb_pept = self.esm2(tokens_pept).hidden  # ([batch], seq_length, 320)
        emb_pept = jnp.transpose(emb_pept, (1, 0))  # out: ([batch], 320, seq_length)
        x = jnp.array(emb_pept)

        # protein cnn stack
        for layer in self.cnn_stack:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x, key= key)
            else:
                x = layer(x)


        # simple mean along embedding, however could be extended to attention_mean, not sure if necessary
        x_pooled = jnp.mean(
            x, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)

        x_pooled = jnp.transpose(x_pooled, (1, 0))  # out: ([batch], 1, embedding)
        x_peptide = x_pooled.squeeze()  # out: ([batch], embedding)

        ### PREDICTION-HEAD ###

        conc = jnp.concat((x_protein, x_peptide))  # out: ([batch], embedding * 2)
        pred_aff = self.prediction_head(conc, key)

        return pred_aff
