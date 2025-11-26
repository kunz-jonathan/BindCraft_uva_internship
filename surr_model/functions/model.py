import equinox as eqx
import esm2quinox
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable
import optax 


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
        key1, key2, key3 = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_features=128, out_features=64, key=key1),
            #jax.nn.relu,
            eqx.nn.Dropout(p=0.2),
            eqx.nn.Linear(in_features=64, out_features=32, key=key2),
            #jax.nn.hard_tanh,
            eqx.nn.Dropout(p=0.2),
            eqx.nn.Linear(in_features=32, out_features=1, key=key3),
        ]

    def __call__(self, x, key):
        # x = jnp.concat((tokens_prot, tokens_pept))  # [H] -> [2H]
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x, key=key)
            else:
                x = layer(x)
        return x



class AFF_PREDICTOR(eqx.Module):
    esm2: esm2quinox.ESM2
    cnn_stack: list[eqx.nn.Conv1d | eqx.nn.Dropout | eqx.nn.BatchNorm | Callable]
    prediction_head: Prediction_Head

    def __init__(self, model, key):
        key1, key2, key3, key4 = jr.split(key, 4)

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
            jax.nn.relu,
            eqx.nn.Dropout(p=0.2),
            eqx.nn.BatchNorm(256, axis_name="batch"),
            eqx.nn.Conv1d(
                in_channels=256,
                out_channels=128,
                kernel_size=5,
                padding="SAME",
                stride=1,
                key=key2,
            ),
            jax.nn.relu,
            eqx.nn.Dropout(p=0.2),
            eqx.nn.BatchNorm(128, axis_name="batch"),
            eqx.nn.Conv1d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                padding="SAME",
                stride=1,
                key=key3,
            ),
            jax.nn.relu,
            eqx.nn.Dropout(p=0.2),
            eqx.nn.BatchNorm(64, axis_name="batch"),
        ]

        self.prediction_head = Prediction_Head(key4)

    def __call__(self, tokens_prot, tokens_pept, state, key):
        ### PROTEIN ###
        emb_prot = self.esm2(tokens_prot).hidden  # ([batch], seq_length, 320)
        emb_prot = jnp.transpose(emb_prot, (1, 0))  # out: ([batch], 320, seq_length)
        x = jnp.array(emb_prot)

        for layer in self.cnn_stack:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x, key=key)
            elif isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)

        # simple mean along embedding, however could be extended to attention_mean, not sure if necessary
        x_pooled = jnp.mean(
            x, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)

        x_protein = jnp.transpose(
            x_pooled, (1, 0)
        ).squeeze()  # out: ([batch], embedding)

        ### PEPTIDE ###
        emb_pept = self.esm2(tokens_pept).hidden  # ([batch], seq_length, 320)
        emb_pept = jnp.transpose(emb_pept, (1, 0))  # out: ([batch], 320, seq_length)
        x = jnp.array(emb_pept)

        # protein cnn stack
        for layer in self.cnn_stack:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x, key=key)
            elif isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)

        # simple mean along embedding, however could be extended to attention_mean, not sure if necessary
        x_pooled = jnp.mean(
            x, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)

        x_peptide = jnp.transpose(
            x_pooled, (1, 0)
        ).squeeze()  # out: ([batch], embedding)

        ### PREDICTION-HEAD ###

        conc = jnp.concat((x_protein, x_peptide))  # out: ([batch], embedding * 2)
        pred_aff = self.prediction_head(conc, key)

        return pred_aff, state

class stripped_PREDICTOR(eqx.Module):
    esm2: esm2quinox.ESM2
    prot_droput: eqx.nn.Dropout
    pept_droput: eqx.nn.Dropout
    prot_projection: eqx.nn.Linear
    pept_projection: eqx.nn.Linear
    

    def __init__(self, model, key):
        key1, key2= jr.split(key, 2)

        self.esm2 = model  # (num_layers=3, embed_size=32, num_heads=2, token_dropout=False, key=key)
        # output size is 480
        self.prot_droput = eqx.nn.Dropout(p=0.2)
        self.prot_projection = eqx.nn.Linear(in_features=640,out_features=320,key=key1)
        
        self.pept_droput = eqx.nn.Dropout(p=0.2)
        self.pept_projection = eqx.nn.Linear(in_features=640,out_features=320,key=key2)

    def __call__(self, tokens_prot, tokens_pept, state, key):
        ### PROTEIN ###
        emb_prot = self.esm2(tokens_prot).hidden  # ([batch], seq_length, 320)
        emb_prot = jnp.transpose(emb_prot, (1, 0))  # out: ([batch], 320, seq_length)
        x_prot = jnp.array(emb_prot)
        x_prot = jnp.mean(
            x_prot, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)
        x_prot = jnp.transpose(
            x_prot, (1, 0)
        ).squeeze() 
        x_prot = self.prot_droput(x_prot,key=key)
        
        x_prot = self.prot_projection(x_prot)
        

        ### PEPTIDE ###
        emb_pept = self.esm2(tokens_pept).hidden  # ([batch], seq_length, 320)
        emb_pept = jnp.transpose(emb_pept, (1, 0))  # out: ([batch], 320, seq_length)
        x_pept = jnp.array(emb_pept)
        x_pept = jnp.mean(
            x_pept, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)
        x_pept = jnp.transpose(
            x_pept, (1, 0)
        ).squeeze() 
        x_pept = self.pept_droput(x_pept,key=key)
        x_pept = self.pept_projection(x_pept)


        ### PREDICTION-HEAD ###
        pred_aff = optax.cosine_similarity(x_prot,x_pept)
        

        return pred_aff, state
    
    
class fine_tune_PREDICTOR(eqx.Module):
    esm2: esm2quinox.ESM2
    prot_droput: eqx.nn.Dropout
    pept_droput: eqx.nn.Dropout
    prot_projection: eqx.nn.Linear
    pept_projection: eqx.nn.Linear
    prot_inter_one: eqx.nn.Linear
    pept_inter_one: eqx.nn.Linear
    prot_inter_drop: eqx.nn.Dropout
    pept_inter_drop: eqx.nn.Dropout
    

    def __init__(self, model, key):
        key1, key2,key3,key4= jr.split(key, 4)

        self.esm2 = model  # (num_layers=3, embed_size=32, num_heads=2, token_dropout=False, key=key)
        # output size is 480
        self.prot_droput = eqx.nn.Dropout(p=0.2)
        self.prot_projection = eqx.nn.Linear(in_features=640,out_features=480,key=key1)
        
        self.pept_droput = eqx.nn.Dropout(p=0.2)
        self.pept_projection = eqx.nn.Linear(in_features=640,out_features=480,key=key2)
        
        self.prot_inter_drop = eqx.nn.Dropout(p=0.2)
        self.prot_inter_one = eqx.nn.Linear(in_features=480,out_features=320,key=key3)
        
        self.pept_inter_drop = eqx.nn.Dropout(p=0.2)
        self.pept_inter_one = eqx.nn.Linear(in_features=480,out_features=320,key=key4)

    def __call__(self, tokens_prot, tokens_pept, state, key):
        ### PROTEIN ###
        emb_prot = self.esm2(tokens_prot).hidden  # ([batch], seq_length, 320)
        emb_prot = jnp.transpose(emb_prot, (1, 0))  # out: ([batch], 320, seq_length)
        x_prot = jnp.array(emb_prot)
        x_prot = jnp.mean(
            x_prot, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)
        x_prot = jnp.transpose(
            x_prot, (1, 0)
        ).squeeze() 
        x_prot = self.prot_droput(x_prot,key=key)
        
        x_prot = self.prot_inter_one(x_prot)
        x_prot = self.prot_inter_drop(x_prot,key=key)
        
        x_prot = self.prot_projection(x_prot)
        

        ### PEPTIDE ###
        emb_pept = self.esm2(tokens_pept).hidden  # ([batch], seq_length, 320)
        emb_pept = jnp.transpose(emb_pept, (1, 0))  # out: ([batch], 320, seq_length)
        x_pept = jnp.array(emb_pept)
        x_pept = jnp.mean(
            x_pept, axis=1, keepdims=True
        )  # mean along the sequence so that out: ([batch],embedding,1)
        x_pept = jnp.transpose(
            x_pept, (1, 0)
        ).squeeze() 
        x_pept = self.pept_droput(x_pept,key=key)
        
        x_pept = self.pept_inter_one(x_pept)
        x_pept = self.pept_inter_drop(x_pept,key=key)

        x_pept = self.pept_projection(x_pept)


        ### PREDICTION-HEAD ###
        pred_aff = optax.cosine_similarity(x_prot,x_pept)
        

        return pred_aff, state

