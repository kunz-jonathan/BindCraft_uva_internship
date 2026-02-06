import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
import equinox as eqx
import esm  # pip install fair-esm==2.0.0
import esm2quinox
import jax.random as jrandom
import optax  # pip install optax
from dataset.dataset import Dataset_PEPBI
from functions.model import (
    stripped_PREDICTOR,
)
from functions.training import (
    eval_step,
    train_model_validation,
)
import jax.numpy as jnp
import jax.random as jr
import jax
from torch.utils.data import DataLoader
import pickle


base_path = "/home/kunzj"

data_ppi_train_three_split = Dataset_PEPBI(
    columns=["Prot_Seq", "Pept_Seq", "Energy"],
    data_path=f"{base_path}/BindCraft_uva_internship/data/ppi_affinity_dataset/cosine_sim_scaled/three_way/train.csv",
    transform=esm2quinox.tokenise,
)
data_ppi_validation_three_split = Dataset_PEPBI(
    columns=["Prot_Seq", "Pept_Seq", "Energy"],
    data_path=f"{base_path}/BindCraft_uva_internship/data/ppi_affinity_dataset/cosine_sim_scaled/three_way/validation.csv",
    transform=esm2quinox.tokenise,
)
data_ppi_test_three_split = Dataset_PEPBI(
    columns=["Prot_Seq", "Pept_Seq", "Energy"],
    data_path=f"{base_path}/BindCraft_uva_internship/data/ppi_affinity_dataset/cosine_sim_scaled/three_way/test.csv",
    transform=esm2quinox.tokenise,
)


loader_ppi_train_three_split = DataLoader(data_ppi_train_three_split, batch_size=3)
loader_ppi_validation_three_split = DataLoader(
    data_ppi_validation_three_split, batch_size=1
)
loader_ppi_test_three_split = DataLoader(data_ppi_test_three_split, batch_size=1)


# generating keys
model_key, call_key = jr.split(jrandom.PRNGKey(0), 2)

# initializing models
torch_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
model_esm2 = esm2quinox.from_torch(torch_model)

model_aff = stripped_PREDICTOR(
    model= model_esm2, key=model_key
)

# initializing optimizer
optim = optax.adam(learning_rate=0.001)
opt_state = optim.init(eqx.filter(model_aff, eqx.is_inexact_array))
best_model_first, train_losses_first, val_losses_first = (
    train_model_validation(
        training_DataLoader=loader_ppi_train_three_split,
        validation_Dataloader=loader_ppi_validation_three_split,
        max_epochs=30,
        model_aff=model_aff,
        opt_state=opt_state,
        optim=optim,
        key=call_key,
    )
)

test_key_init = jrandom.PRNGKey(1)
inference_model = eqx.nn.inference_mode(best_model_first)

test_loss_list = []
pred_y_list = []
y_val_list = []

for entry, (x_prot, x_pept, y_val) in enumerate(loader_ppi_test_three_split):
    x_prot, x_pept, y_val = jnp.array(x_prot), jnp.array(x_pept), jnp.array(y_val)
    test_key = jr.split(test_key_init, x_pept.shape[0])
    test_loss, pred_y = eval_step(inference_model, x_prot, x_pept, y_val, test_key)
    test_loss_list.append(test_loss.item())
    pred_y_list.append(pred_y.item())
    y_val_list.append(y_val.item())
    print(
        f"[Entry {entry + 1}] , Test Loss: {test_loss:.3f}, Predicted Y: {pred_y.item():.3f}, , True Y: {y_val.item():.3f}"
    )

results_ppi_three_way = {
    "test_loss_list": test_loss_list,
    "pred_y_list": pred_y_list,
    "y_val_list": y_val_list,
    "train_losses": train_losses_first,
    "val_losses": val_losses_first,
}

with open("/home/kunzj/BindCraft_uva_internship/surr_model/training_results_jax/results_ppi_three_way.pkl", "wb") as f:
    pickle.dump(results_ppi_three_way, f)
