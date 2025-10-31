import equinox as eqx
import esm  # pip install fair-esm==2.0.0
import esm2quinox
import jax.random as jrandom
import optax  # pip install optax
from dataset import initialize_datasets, Dataset_PEPBI
from model import AFF_PREDICTOR
from training_wo_val import train_model, eval_step
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.random as jr
import jax
import pandas as pd
from torch.utils.data import DataLoader


data_train = Dataset_PEPBI(
    columns=["Prot_Seq", "Pept_Seq", "Energy"],
    data_path="/home/kunzj/dataset/pepbi-dataset/train_data_pepbi.csv",
    transform=esm2quinox.tokenise
)

train_loader = DataLoader(data_train,batch_size=3)


# generating keys
model_key, call_key = jr.split(jrandom.PRNGKey(0), 2)

# initializing models
torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
model_esm2 = esm2quinox.from_torch(torch_model)
model_aff, state = eqx.nn.make_with_state(AFF_PREDICTOR)(
    model=model_esm2, key=model_key
)

# initializing optimizer
optim = optax.adam(learning_rate=0.001)
opt_state = optim.init(eqx.filter(model_aff, eqx.is_inexact_array))
call_key = jr.split(call_key,3)

# train model
best_model, train_losses = train_model(
    training_DataLoader=train_loader,
    max_epochs=30,
    model_aff=model_aff,
    key=call_key,
    opt_state=opt_state,
    optim=optim,
)

eqx.tree_serialise_leaves('/home/kunzj/testing/model_Training/model.eqx', best_model)