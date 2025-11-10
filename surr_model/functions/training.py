import equinox as eqx
import jax
import jax.numpy as jnp
import tqdm
import jax.tree_util as jtu

def filter_model(model):
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
                lambda t, i=i: (t.prediction_head.layers[i].weight, t.prediction_head.layers[i].bias),
                filter_spec,
                replace=(True, True),
            )
    
    return filter_spec

@eqx.filter_value_and_grad(has_aux=True)
def compute_loss(diff_model, static_model , model_state, x_prot, x_pept, y, key):
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
    loss = jnp.mean((pred_y - y) ** 2)

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
    loss = jnp.mean((pred_y - y) ** 2)
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
    filter_spec= filter_model(model)
    diff_model, static_model = eqx.partition(model, filter_spec)
    (loss, model_state), grads = compute_loss(
        diff_model, static_model , model_state, x_prot, x_pept, y, key
    )
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, model_state, opt_state


def train_model(
    training_DataLoader, max_epochs, model_aff, model_state, opt_state, optim, key
):
    """g
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
    return (
        model_aff,
        train_losses,
        model_state
    )
