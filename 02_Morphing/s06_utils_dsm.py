import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tqdm
from flax.training import train_state
from jax.flatten_util import ravel_pytree

import s01_utils as utils
import s03_models as models

SIGMAS = utils.get_sigmas()


# Annealed Langevin sampling
def sample_annealed_langevin(apply_fn, x_initial, params, key, eps=2e-5, num_steps=100):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    x_t, x_sequence = x_initial, [x_initial]

    def _outer_step(xt, args):
        i, key = args
        step_size = eps * SIGMAS[i] ** 2 / SIGMAS[-1] ** 2

        def _inner_step(xt, key):
            key, subkey = jax.random.split(key)
            z_t = jr.normal(subkey, shape=xt.shape)
            xtt = xt + step_size * apply_fn(params, xt, jnp.array([i])) + jnp.sqrt(2 * step_size) * z_t

            return xtt, xtt

        keys = jr.split(key, num_steps)
        _, xts = jax.lax.scan(_inner_step, xt, keys)

        return xts[-1], xts

    keys = jr.split(key, len(SIGMAS))
    _, x_sequence = jax.lax.scan(_outer_step, x_t, (jnp.arange(len(SIGMAS)), keys))
    x_sequence = jnp.concatenate(x_sequence, axis=0)

    return x_sequence


def denoising_loss_fn(params, X, y, applyfn, key):
    X, y = X[None, ...], y[None, ...]
    sigma_curr = SIGMAS[y]
    X_perturbed = X + sigma_curr * jr.normal(key, X.shape)
    target = -(X_perturbed - X) / sigma_curr**2
    score = applyfn(params, X_perturbed, y)
    loss = 1 / 2.0 * ((score - target) ** 2).sum() * (sigma_curr**2)

    return jnp.mean(loss)


@jax.jit
def train_step(state, X_batch, y_batch, key):
    """Train for a single step."""
    v_loss = lambda params, xs, ys, keys: jnp.mean(
        jax.vmap(denoising_loss_fn, in_axes=(None, 0, 0, None, 0))(params, xs, ys, state.apply_fn, keys)
    )
    keys = jr.split(key, len(X_batch))
    loss = v_loss(state.params, X_batch, y_batch, keys)
    grads = jax.grad(v_loss)(state.params, X_batch, y_batch, keys)
    state = state.apply_gradients(grads=grads)

    return state, loss


def create_train_state(flat_params, apply_fn, learning_rate):
    """Creates initial `TrainState`."""
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=apply_fn, params=flat_params, tx=tx)

    return state


def train_epoch(state, X, key, batch_size):
    key, subkey = jr.split(key)
    idx = jr.permutation(key, jnp.arange(X.shape[0]))
    X = X[idx]
    n_batches = len(X) // batch_size
    loss_history = []

    for i in range(n_batches):
        key1, key2, subkey = jr.split(subkey, 3)
        X_curr = X[i * batch_size : (i + 1) * batch_size]
        y_curr = jr.randint(key1, (batch_size,), 0, len(SIGMAS))
        state, loss = train_step(state, X_curr, y_curr, key2)
        loss_history.append(loss.item())

    return state, loss_history


def train_loop(flat_params, apply_fn, X, lr=1e-3, key=0, batch_size=128, n_epochs=1_000, verbose=True):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)
    state = create_train_state(flat_params, apply_fn, lr)
    loss_history = []
    if verbose:
        pbar = tqdm.tqdm(range(n_epochs), desc="Loss: 0.0")
    else:
        pbar = range(n_epochs)
    for i in pbar:
        key, subkey = jr.split(subkey)
        state, losses = train_epoch(state, X, key, batch_size)
        loss_history.extend(losses)
        if verbose:
            pbar.set_description(f"Loss: {jnp.mean(jnp.array(loss_history[-20:])):.3f}")
    loss_history = jnp.array(loss_history)

    return state, loss_history


def train_dsm(X, key=0, n_epochs=100, batch_size=128, n_features=16, lr=5e-4):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    key, subkey = jr.split(key)

    model = models.NCSN(num_features=n_features)
    params = model.init(key, X[0:1], jnp.array([0]))
    flat_params, unflatten_fn = ravel_pytree(params)
    print(f"Number of parameters: {len(flat_params):,}")

    apply_fn = lambda flat_params, x, y: model.apply(unflatten_fn(flat_params), x, y)
    key, subkey = jr.split(subkey)

    state, loss_history = train_loop(flat_params, apply_fn, X, lr=lr, key=key, batch_size=batch_size, n_epochs=n_epochs)

    return state, loss_history
