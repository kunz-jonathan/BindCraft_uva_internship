import equinox as eqx
import jax
import jax.numpy as jnp
import tqdm
import jax.tree_util as jtu
import jax.random as jr


def filter_model(model):
    """
    freezes the esm2 stack in the model
    """
    filter_spec = jtu.tree_map(lambda _: False, model)

    # CNN stack
    for i, layer in enumerate(model.cnn_stack):
        if isinstance(layer, (eqx.nn.Conv1d, eqx.nn.BatchNorm)):
            filter_spec = eqx.tree_at(
                lambda t, i=i: (t.cnn_stack[i].weight, t.cnn_stack[i].bias),
                filter_spec,
                replace=(True, True),
            )

    # Prediction head
    for i, layer in enumerate(model.prediction_head.layers):
        if isinstance(layer, eqx.nn.Linear):
            filter_spec = eqx.tree_at(
                lambda t, i=i: (
                    t.prediction_head.layers[i].weight,
                    t.prediction_head.layers[i].bias,
                ),
                filter_spec,
                replace=(True, True),
            )

    return filter_spec

def filter_model_stripped(model):
    """
    Freeze the esm2 stack in the stripped_PREDICTOR model and
    enable training for the two projection Linear layers.
    """
    # start with everything frozen
    filter_spec = jtu.tree_map(lambda _: False, model)

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


@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(diff_model, static_model, model_state, x_prot, x_pept, y, key):
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

    pred_y, model_state = jax.vmap(
        model, axis_name="batch", in_axes=(0, 0, None), out_axes=(0, None)
    )(x_prot, x_pept, model_state, key=key)
    loss = jnp.mean(jnp.abs(pred_y - y))

    return loss, model_state


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
    pred_y, _ = jax.vmap(model)(
        x_prot, x_pept, key=key
    )  # do i actually need the axis_stuff
    loss = jnp.mean(jnp.abs(pred_y - y) )
    return loss, pred_y


@eqx.filter_jit
def make_step(model, model_state, x_prot, x_pept, y, opt_state, optim, key):
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
    filter_spec = filter_model(model)
    diff_model, static_model = eqx.partition(model, filter_spec)
    (loss, model_state), grads = compute_loss(
        diff_model, static_model, model_state, x_prot, x_pept, y, key
    )
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, model_state, opt_state

@eqx.filter_jit
def make_step_stripped(model, model_state, x_prot, x_pept, y, opt_state, optim, key):
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
    (loss, model_state), grads = compute_loss(
        diff_model, static_model, model_state, x_prot, x_pept, y, key
    )
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, model_state, opt_state


def train_model(
    training_DataLoader, max_epochs, model_aff, model_state, optim, opt_state, key
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

    Returns:
        model_aff
        model_state
        train_losses

    """

    train_losses = []

    # use this when model is actually working
    # best_state = eqx.tree_serialise_leaves(state)
    # best_model = eqx.tree_serialise_leaves(model)
    for epoch in tqdm.tqdm(range(max_epochs), desc="Epochs", position=0, leave=True):
        # ---- TRAIN ----
        for x_prot, x_pept, y in tqdm.tqdm(
            training_DataLoader, desc="Training-Set", position=1, leave=False
        ):
            x_prot, x_pept, y = jnp.array(x_prot), jnp.array(x_pept), jnp.array(y)
            
            loss, model_aff, model_state, opt_state = make_step(
                model_aff, model_state, x_prot, x_pept, y, opt_state, optim, key
            )
            
            train_losses.append(loss.item())

    print("Training complete.")
    
    return (model_aff, model_state, train_losses)


def train_model_validation(
    training_DataLoader,
    validation_Dataloader,
    max_epochs,
    model_aff,
    model_state,
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
    best_state = 0
    best_model = 0

    # use this when model is actually working
    # best_state = eqx.tree_serialise_leaves(state)
    # best_model = eqx.tree_serialise_leaves(model)

    for epoch in tqdm.tqdm(range(max_epochs), desc="Epochs", position=0, leave=False):
        # ---- TRAIN ----
        for x_prot, x_pept, y in tqdm.tqdm(
            training_DataLoader, desc="Training-Set", position=1, leave=False
        ):
            x_prot, x_pept, y = jnp.array(x_prot), jnp.array(x_pept), jnp.array(y)

            loss, model_aff, model_state, opt_state = make_step_stripped(
                model_aff, model_state, x_prot, x_pept, y, opt_state, optim, key
            )

            train_losses.append(loss.item())

        # ---- VALIDATION ----
        inference_model = eqx.nn.inference_mode(model_aff)
        inference_model = eqx.Partial(inference_model, state=model_state)

        val_batch_losses = []

        for x_prot_val, x_pept_val, y_val in tqdm.tqdm(
            validation_Dataloader, desc="Validation-Set", position=2, leave=False
        ):
            x_prot_val, x_pept_val, y_val = (
                jnp.array(x_prot_val),
                jnp.array(x_pept_val),
                jnp.array(y_val),
            )

            test_key = jr.split(key, x_pept_val.shape[0])

            val_loss, _ = eval_step(
                inference_model, x_prot_val, x_pept_val, y_val, test_key
            )

            val_batch_losses.append(val_loss.item())

        val_loss = jnp.mean(val_batch_losses)
        val_losses.append(val_loss)

        print(
            f"[Epoch {epoch + 1}] Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_loss:.6f}"
        )

        # ---- Checkpoint Best ----
        if val_loss < best_val:
            best_val = val_loss
            best_state = model_state
            best_model = model_aff

            # best_state = eqx.tree_serialise_leaves(state)
            # best_model = eqx.tree_serialise_leaves(model)
            print("\t(New best model saved.)")
    print("Training complete.")
    return (
        best_model,
        best_state,
        train_losses,
        val_losses,
    )
