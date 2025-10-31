import equinox as eqx
import jax
import jax.numpy as jnp
import tqdm


@eqx.filter_value_and_grad()
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
    pred_y = jax.vmap(model)(
        x_prot, x_pept,key=key
    )
    print(pred_y.shape)
    loss = jnp.mean((pred_y - y) ** 2)

    return loss


@eqx.filter_jit
def eval_step(model, x_prot,x_pept, y,key):
    """
    Evaluation step (Validation, Test) with vmap across first axis

    Args:
        model
        x,y

    Returns:
        MSE
        pred_Y
    """
    pred_y, _ = jax.vmap(model)(x_prot,x_pept,key=key)  # do i actually need the axis_stuff
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
    loss, grads = compute_loss(model, x_prot, x_pept, y, key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_model(
    training_DataLoader,
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


    print("Training complete.")
    return (
        model_aff,
        train_losses,
    )
