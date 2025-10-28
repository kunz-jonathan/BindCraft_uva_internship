import equinox as eqx
import jax
import jax.numpy as jnp
import tqdm


@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(model, x_prot, x_pept, y, key):
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
    pred_y = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(
        x_prot, x_pept,key=key
    )
    loss = jnp.mean((pred_y - y) ** 2)

    return loss


@eqx.filter_jit
def eval_step(model, x, y):
    """
    Evaluation step (Validation, Test) with vmap across first axis

    Args:
        model
        x,y

    Returns:
        MSE
        pred_Y
    """
    pred_y, _ = jax.vmap(model)(x)  # do i actually need the axis_stuff
    loss = jnp.mean((pred_y - y) ** 2)
    return loss, pred_y


@eqx.filter_jit
def make_step(model, x_prot, x_pept, y, opt_state, optim, key):
    """
    Gradient update step

    Args:
        model
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
    (loss, state), grads = compute_loss(model, x_prot, x_pept, y, key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model(
    training_DataLoader,
    validation_Dataloader,
    max_epochs,
    model_aff,
    key,
    opt_state,
    optim,
):
    """
    training wrapper

    Args:
        test_dataloader
        training_dataloder
        initialized model
        initialized model state
        initialized optim state
        num_echos to train


    """

    train_losses = []
    val_losses = []
    best_val = jnp.inf

    best_model = 0

    # use this when model is actually working
    # best_state = eqx.tree_serialise_leaves(state)
    # best_model = eqx.tree_serialise_leaves(model)
    for epoch in tqdm.tqdm(range(max_epochs), desc="Epochs", position=0, leave=True):
        # ---- TRAIN ----
        for x_prot, x_pept, y in tqdm.tqdm(
            training_DataLoader, desc="Training-Set", position=1, leave=False
        ):
            x_prot, x_pept, y = jnp.array(x_prot), jnp.array(x_pept), jnp.array(y)
            loss, model_aff, opt_state = make_step(
                model_aff, x_prot, x_pept, y, opt_state, optim, key
            )
        train_losses.append(loss.item())

        # ---- VALIDATION ----
        inference_model = eqx.nn.inference_mode(model_aff)
        inference_model = eqx.Partial(inference_model)
        val_batch_losses = []
        for x_val, y_val in tqdm.tqdm(
            validation_Dataloader, desc="Validation-Set", position=2, leave=False
        ):
            x_val, y_val = jnp.array(x_val), jnp.array(y_val)
            val_loss, _ = eval_step(inference_model, x_val, y_val)
            val_batch_losses.append(val_loss.item())

        val_loss = jnp.mean(jnp.array(val_batch_losses))
        val_losses.append(val_loss)
        print(
            f"[Epoch {epoch + 1}] Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}"
        )

        # ---- Checkpoint Best ----
        if val_loss < best_val:
            best_val = val_loss
            best_model = model_aff

            # best_state = eqx.tree_serialise_leaves(state)
            # best_model = eqx.tree_serialise_leaves(model)
            print("\t(New best model saved.)")
    print("Training complete.")
    return (
        best_model,
        train_losses,
        val_losses,
    )
