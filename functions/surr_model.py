import jax
import jax.numpy as jnp
import haiku as hk
import optax

import pandas as pd
import pickle

from jax import config

config.update("jax_default_matmul_precision", "float32")



class CNNAffinityPredictor(hk.Module):
    def __init__(self, seq_hidden=128, prop_hidden=32, fc_hidden=[64, 32], name=None):
        super().__init__(name=name)
        self.seq_hidden = seq_hidden
        self.prop_hidden = prop_hidden
        self.fc_hidden = fc_hidden

    def __call__(self, seq_onehot ):
        """
        seq_onehot: [batch_size, seq_len, num_amino_acids]
        """
        # ---- Sequence branch: CNN ----
        x = seq_onehot  # [B, L, A]

        # First 1D convolution (along sequence length)
        conv1 = hk.Conv1D(output_channels=64, kernel_shape=3, stride=1, padding="SAME")
        x = conv1(x)
        x = jax.nn.relu(x)

        # Second 1D convolution
        conv2 = hk.Conv1D(
            output_channels=self.seq_hidden, kernel_shape=3, stride=1, padding="SAME"
        )
        x = conv2(x)
        x = jax.nn.relu(x)

        # Third 1D convolution
        conv3 = hk.Conv1D(
            output_channels=self.seq_hidden, kernel_shape=3, stride=1, padding="SAME"
        )
        x = conv3(x)
        x = jax.nn.relu(x)

        # Global mean pooling along sequence length
        seq_vector = jnp.mean(x, axis=1)  # [B, seq_hidden]
        
        # ---- Fully connected layers ----
        fc = seq_vector
        for hidden in self.fc_hidden:
            fc = hk.Linear(hidden)(fc)
            fc = jax.nn.relu(fc)

        # ---- Output layer ----
        affinity = hk.Linear(1)(fc)  # scalar output
        return affinity


# Haiku transform
def model_fn(seq_onehot ):
    model = CNNAffinityPredictor()
    return model(seq_onehot )


# ---- actual data ----

# TO DO - this is actually also alr. in colabfold ...
# from colabdesign.af.alphafold.common.residue_constants import * 
def pad_sequences(sequences, max_len=15, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """
    sequences: list of sequences of different lengths, each as a string (e.g. 'ACDE')
    max_len: length to pad/truncate sequences to
    alphabet: string of possible amino acids (default 20 canonical)

    returns: [batch, max_len, num_amino_acids] one-hot encoded
    """

    # map amino acid letters to indices
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    num_amino_acids = len(alphabet)

    batch_size = len(sequences)
    padded = jnp.zeros((batch_size, max_len), dtype=jnp.int32)

    for i, seq in enumerate(sequences):
        indices = seq
        if type(seq[0]) is not int:  # if input is already indices
            indices = [aa_to_idx.get(aa, 0) for aa in seq]  # default to 0 if unknown
        # print(')

        padded = padded.at[i, : len(indices)].set(jnp.array(indices, dtype=jnp.int32))
    # one-hot encode
    onehot = jax.nn.one_hot(padded, num_classes=num_amino_acids, dtype=jnp.float32)
    return onehot


# ---- Loss and training step ----
def loss_fn(params, seq_onehot , targets):
    preds = model.apply(params, rng, seq_onehot )
    return jnp.mean((preds - targets) ** 2)


@jax.jit
def train_step(params, opt_state, seq_onehot , targets):
    grads = jax.grad(loss_fn)(params, seq_onehot , targets)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    loss = loss_fn(params, seq_onehot, targets)
    return params, opt_state, loss


if __name__ == "__main__":
    # ---- model ----
    storage = input('store model weights (yes:smth./no:Enter): ')

    model = hk.transform(model_fn)

    # ---- data ----
    df = pd.read_csv("/home/kunzj/bindcraft_modified/flexs/landscape.csv")

    seq_len = 15
    num_amino_acids = 20

    sequences = df["sequence"].tolist()
    seq_onehot = pad_sequences(sequences=sequences)
    batch_size = len(sequences)
    rng = jax.random.PRNGKey(42)

    targets = jnp.array(df["mmpbsa"].values, dtype=jnp.float32)

    # ---- initialization ----
    params = model.init(rng=rng, seq_onehot=seq_onehot)

    # params = jtu.tree_map(lambda x: x.astype(jnp.float64), params)

    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # ---- Example training loop ----
    losses = []
    for step in range(501):
        params, opt_state, loss = train_step(
            params, opt_state, seq_onehot , targets
        )
        losses.append(loss)

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
    if storage:
        with open("/home/kunzj/BindCraft_uva_internship/surr_model_params/params_only_seq.pkl", "wb") as f:
            pickle.dump(params,f)


def init_surr_model():
    """
    reads MD_data
    initializes model

    Returns:
        model (class?): initialized model
        params_restored (dic?): trained parameter of model
        rng (): key
    """
    model = hk.transform(model_fn)

    # ---- data ----
    df = pd.read_csv("/home/kunzj/bindcraft_modified/flexs/landscape.csv")


    sequences = df["sequence"].tolist()
    seq_onehot = pad_sequences(sequences=sequences)

    rng = jax.random.PRNGKey(42)

    model.init(rng, seq_onehot )

    print("start loading miodel weights")
    with open("/home/kunzj/BindCraft_uva_internship/surr_model_params/params_only_seq.pkl", "rb") as f:
        params_restored = pickle.load(f)

    return model, params_restored, rng
