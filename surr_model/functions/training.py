import equinox as eqx
import jax
import jax.numpy as jnp
import tqdm
import jax.tree_util as jtu
import jax.random as jr


def filter_model_stripped(model):
    """
    Freeze the esm2 stack in the stripped_PREDICTOR model and
    enable training for the two projection Linear layers.
    """
    # start with everything frozenFalse
    filter_spec = jtu.tree_map(lambda _: False, model)

    filter_spec = eqx.tree_at(
        lambda t: (t.pooler_layer_prot.weight, t.pooler_layer_prot.bias),
        filter_spec,
        replace=(True, True),
    )

    filter_spec = eqx.tree_at(
        lambda t: (t.pooler_layer_pept.weight, t.pooler_layer_pept.bias),
        filter_spec,
        replace=(True, True),
    )
    
    # Unfreeze prot_projection weight & bias
    filter_spec = eqx.tree_at(
        lambda t: (t.prot_projection.weight, t.prot_projection.bias),
        filter_spec,
        replace=(True, True),
    )

    # Unfreeze pept_projection weight & bias
    filter_spec = eqx.tree_at(
        lambda t: (t.pept_projection.weight, t.pept_projection.bias),
        filter_spec,
        replace=(True, True),
    )

    return filter_spec


@eqx.filter_value_and_grad()
def compute_loss(diff_model, static_model, x_prot, x_pept, y, key):
    """
    Computes loss batched with vmap across first axis

    Args:
        model
        state
        x,y

    Returns:
        MSE
        updated model state
    """
    key = jax.random.split(key, x_prot.shape[0])

    model = eqx.combine(diff_model, static_model)

    pred_y = jax.vmap(model)(x_prot, x_pept, key=key)
    loss = jnp.mean((y - pred_y) ** 2)

    return loss


@eqx.filter_jit
def eval_step(model, x_prot, x_pept, y, key):
    """
    Evaluation step (Validation, Test) with vmap across first axis

    Args:
        model
        x,y

    Returns:
        MSE
        pred_Y
    """
    pred_y = jax.vmap(model)(
        x_prot, x_pept, key=key
    )  
    loss = jnp.mean((y - pred_y) ** 2)
    return loss, pred_y


@eqx.filter_jit
def make_step_stripped(model, x_prot, x_pept, y, opt_state, optim, key):
    """
    Gradient update step

    Args:
        modelfilter_spec
        state
        x,y
        optimizer state
        model satte
        optimizer

    Returns:
        MSE-loss
        updated model
        updated state
        updated optimizer state
    """
    filter_spec = filter_model_stripped(model)
    diff_model, static_model = eqx.partition(model, filter_spec)
    loss, grads = compute_loss(
        diff_model, static_model, x_prot, x_pept, y, key
    )
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model_validation(
    training_DataLoader,
    validation_Dataloader,
    max_epochs,
    model_aff,
    optim,
    opt_state,
    key,
):
    """
    training wrapper with validation step

    Args:
        test_dataloader
        training_dataloder
        initialized model
        initialized model state
        initialized optim state
        num_echos to train

    Returns:
        best_model
        best_state
        train_losses
        val_losses

    """

    train_losses = []
    val_losses = []
    best_val = jnp.inf

    best_model = 0

    # use this when model is actually working
    # best_state = eqx.tree_serialise_leaves(state)
    # best_model = eqx.tree_serialise_leaves(model)

    for epoch in tqdm.tqdm(range(max_epochs), desc="Epochs", position=0, leave=False):

        train_batch_losses = 0
        # ---- TRAIN ----
        for x_prot, x_pept, y in training_DataLoader:
            x_prot, x_pept, y = jnp.array(x_prot), jnp.array(x_pept), jnp.array(y)

            loss, model_aff, opt_state = make_step_stripped(
                model_aff, x_prot, x_pept, y, opt_state, optim, key
            )

            train_batch_losses += loss.item()

        train_losses.append((train_batch_losses / training_DataLoader.__len__()))
        # ---- VALIDATION ----
        inference_model = eqx.nn.inference_mode(model_aff)


        val_batch_losses = 0

        for x_prot_val, x_pept_val, y_val in validation_Dataloader:
            x_prot_val, x_pept_val, y_val = (
                jnp.array(x_prot_val),
                jnp.array(x_pept_val),
                jnp.array(y_val),
            )

            test_key = jr.split(key, x_pept_val.shape[0])

            val_loss, _ = eval_step(
                inference_model, x_prot_val, x_pept_val, y_val, test_key
            )

            val_batch_losses += val_loss.item()

        val_losses.append((val_batch_losses / validation_Dataloader.__len__()))

        print(
            f"[Epoch {epoch + 1}] Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}"
        )

        # ---- Checkpoint Best ----
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_model = model_aff

            # best_state = eqx.tree_serialise_leaves(state)
            # best_model = eqx.tree_serialise_leaves(model)
            print(f"\t(New best model saved at Epoch: {epoch +1}. )")
    print("Training complete.")
    # eqx.tree_serialise_leaves('/home/kunzj/BindCraft_uva_internship/surr_model/training_results_jax/model_ppi_last_epoch.eqx',model_aff)
    
    return (
        best_model,
        train_losses,
        val_losses,
    )
