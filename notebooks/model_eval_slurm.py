import equinox as eqx
import esm  # pip install fair-esm==2.0.0
import esm2quinox
import jax.random as jrandom
import optax  # pip install optax
from dataset import initialize_datasets, Dataset_PEPBI
from model import AFF_PREDICTOR
from surr_model.functions.training_wo_val import train_model, eval_step
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax.random as jr
import jax
import pandas as pd
from torch.utils.data import DataLoader

data_test = Dataset_PEPBI(
    columns=["Prot_Seq", "Pept_Seq", "Energy"],
    data_path="/home/kunzj/dataset/pepbi-dataset/train_data_pepbi.csv",
    transform=esm2quinox.tokenise
)
test_loader = DataLoader(data_test,batch_size=2)

# generating keys
model_key, call_key = jr.split(jrandom.PRNGKey(0), 2)

# initializing models
torch_model, _ = esm.pretrained.esm2_t6_8M_UR50D()
model_esm2 = esm2quinox.from_torch(torch_model)
model_aff, state = eqx.nn.make_with_state(AFF_PREDICTOR)(
    model=model_esm2, key=model_key
)


newmodel = eqx.tree_deserialise_leaves('/home/kunzj/testing/model_Training/model.eqx', model_aff)

inference_model = eqx.nn.inference_mode(newmodel)

test_loss_list = []

for entry, (x_prot,x_pept, y_val) in enumerate(test_loader):
    x_prot,x_pept, y_val = jnp.array(x_prot),jnp.array(x_pept), jnp.array(y_val)
    test_loss, pred_y = eval_step(inference_model, x_prot,x_pept, y_val,call_key)
    test_loss_list.append(test_loss.item())

    print(
        f"[Entry {entry + 1}] , Test Loss: {test_loss:.6f}, Predicted Y: {pred_y}, , True Y: {y_val}"
    )
