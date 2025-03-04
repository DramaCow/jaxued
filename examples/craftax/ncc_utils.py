import jax
import jax.numpy as jnp
from flax.struct import dataclass
import optax
import chex
from optax._src import numerics


from functools import partial
import operator

"""THE FOLLOWING IS JUST COPIED FROM OPTAX TREE UTILS"""

def tree_sum(tree) -> chex.Numeric:
  """Compute the sum of all the elements in a pytree.

  Args:
    tree: pytree.

  Returns:
    a scalar value.
  """
  sums = jax.tree_util.tree_map(jnp.sum, tree)
  return jax.tree_util.tree_reduce(operator.add, sums, initializer=0)

def _square(leaf):
  return jnp.square(leaf.real) + jnp.square(leaf.imag)

def tree_l2_norm(tree, squared: bool = False) -> chex.Numeric:
  """Compute the l2 norm of a pytree.

  Args:
    tree: pytree.
    squared: whether the norm should be returned squared or not.

  Returns:
    a scalar value.
  """
  squared_tree = jax.tree_util.tree_map(_square, tree)
  sqnorm = tree_sum(squared_tree)
  if squared:
    return sqnorm
  else:
    return jnp.sqrt(sqnorm)

def tree_update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t: (
          (1 - decay) * (g**order) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )

def tree_update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g**order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return numerics.abs_sq(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: (
          (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )



@dataclass
class ScaleByTiAdaState:
    vx: float | None
    vy: float

    # if doing Adam + ScaleByTiAdaState
    prev_grad: dict | jnp.ndarray = None
    exp_b1: float | None = 1.0
    exp_b2: float | None = 1.0

@dataclass 
class ScaleByTiAdaNoAdamState:
    vx: dict[float]
    coord_vx: dict[jnp.ndarray]

    vy: float
    coord_vy: jnp.ndarray = None


def scale_x_by_ti_ada_noadam(
    eta: float = 1e-4,
    alpha: float = 0.6,
):
    def init_fn(params):
        vx = 0.0
        coord_vx = jax.tree_util.tree_map(jnp.zeros_like, params)

        return ScaleByTiAdaNoAdamState(
            vx = vx,
            coord_vx = coord_vx,
            vy = 0.0
        )
    
    def update_fn(x_updates, state, params=None):

        vx = state.vx + tree_l2_norm(x_updates, squared=True)
        coord_vx = jax.tree_util.tree_map(
            lambda prev, curr: prev + jnp.square(curr), state.coord_vx, x_updates
        )
        
        global_coeff = eta * jax.lax.pow(vx, alpha) / jax.lax.pow(jnp.maximum(vx, state.vy) + 1e-6, alpha)
        
        grad = jax.tree_util.tree_map(
            lambda g, c:  g * global_coeff * jax.lax.pow(c + 1e-6, -alpha), x_updates, coord_vx
        )

        state = ScaleByTiAdaNoAdamState(
            vx = vx,
            coord_vx = coord_vx,
            vy = state.vy,
        )

        return grad, state
    
    return optax.GradientTransformation(init_fn, update_fn) 
        
def scale_y_by_ti_ada_noadam(
    eta: float = 1e-4,
    beta: float = 0.4,
):
    def init_fn(params):
        return ScaleByTiAdaNoAdamState(
            vx = None,
            coord_vx = None,
            vy = 0.0,
            coord_vy = jnp.zeros_like(params)
        )
    
    def  update_fn(updates, state, params=None):
        # track vy for the x player's stepsize
        vy = state.vy + jnp.square(jnp.linalg.norm(updates))
        coord_vy = state.coord_vy + jnp.square(updates)

        coeff = eta / jax.lax.pow(coord_vy + 1e-6, beta)

        grad = coeff * updates

        state = ScaleByTiAdaNoAdamState(
            vx = None,
            coord_vx = None,
            vy = vy,
            coord_vy = coord_vy
        )

        return grad, state

    return optax.GradientTransformation(init_fn, update_fn) 

def scale_x_by_ti_ada(
    vx0: float = 0.1,
    vy0: float = 0.1, # just pass in a zeros_like y_params
    eta: float = 1e-4,
    alpha: float = 0.6,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5,
):
    """
    https://openreview.net/pdf?id=zClyiZ5V6sL 
    assumes we are doing the adam version
    """
    def init_fn(params):
        vx = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        prev_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        return ScaleByTiAdaState(vx, vy0, prev_grad = prev_grad)
    
    def update_fn(x_updates, state, params=None):

        grad = tree_update_moment(x_updates, state.prev_grad, b1, 1)
        vx = tree_update_moment_per_elem_norm(x_updates, state.vx, b2, 2)
        
        exp_b1 = state.exp_b1 * b1
        exp_b2 = state.exp_b2 * b2

        total_sum_vx = tree_sum(vx)
        total_sum_vy = state.vy.sum()

        ratio = jax.lax.pow(total_sum_vx, alpha) / jax.lax.pow(jax.lax.max(total_sum_vx, total_sum_vy), alpha)

        coeff = jax.tree_util.tree_map(
            lambda v: eta / (jax.lax.pow(v, alpha) / jnp.sqrt(1 - exp_b2) + eps), vx
        )

        bias_corrected_grad = jax.tree_util.tree_map(lambda m: m / (1 - exp_b1), grad)
    
        x_grad = jax.tree_util.tree_map(
            lambda m, c: ratio * c * m, bias_corrected_grad, coeff
        )

        new_state = ScaleByTiAdaState(
            vx, 
            state.vy, 
            prev_grad=grad,
            exp_b1=exp_b1,
            exp_b2=exp_b2
        )

        return x_grad, new_state

    return optax.GradientTransformation(init_fn, update_fn)

def scale_y_by_ti_ada(
    vy0: float = 0.1, 
    eta: float = 0.1,
    beta: float = 0.40,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5
):
    """
    https://openreview.net/pdf?id=zClyiZ5V6sL 
    assumes we are doing the adam version
    """
    def init_fn(params):
        vx = None
        prev_grad = vy = jnp.zeros_like(params)
        return ScaleByTiAdaState(vx, vy, prev_grad = prev_grad)
    
    def update_fn(y_updates, state, params=None):

        grad = tree_update_moment(y_updates, state.prev_grad, b1, 1)
        vy = tree_update_moment_per_elem_norm(y_updates, state.vy, b2, 2)
        
        exp_b1 = state.exp_b1 * b1
        exp_b2 = state.exp_b2 * b2

        coeff = eta / (jax.lax.pow(vy, beta) / jnp.sqrt(1 - exp_b2) + eps)

        bias_corrected_grad = grad / (1 - exp_b1)
        
        y_grad = bias_corrected_grad * coeff

        new_state = ScaleByTiAdaState(
            None, 
            vy, 
            prev_grad=grad,
            exp_b1=exp_b1,
            exp_b2=exp_b2
        )

        return y_grad, new_state

    return optax.GradientTransformation(init_fn, update_fn)

def ti_ada(
    vx0: float = 0.1,
    vy0: float = 0.1, # just pass in a zeros_like y_params
    eta: float = 1e-4,
    alpha: float = 0.6,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5, 
):
    return optax.chain(
        scale_x_by_ti_ada(vx0, vy0, 1.0, alpha, b1, b2, eps),
        optax.scale(-eta) if isinstance(eta, float) else optax.scale_by_schedule(lambda t: -eta(t)) 
    )

def ti_ada_sgd(
    eta: float = 1e-4,
    alpha: float = 0.6
):
    return optax.chain(
        scale_x_by_ti_ada_noadam(1.0, alpha),
        optax.scale(-eta) if isinstance(eta, float) else optax.scale_by_schedule(lambda t: -eta(t))
    )
    
def projection_simplex_truncated(x: jnp.ndarray, eps: float) -> jnp.ndarray: 
    """
    Code adapted from 
    https://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html
    To represent truncated simplex projection. Assumes 1D vector. 
    """
    ones = jnp.ones_like(x)
    lambdas = jnp.concatenate((ones * eps - x, ones - x), axis=-1)
    idx = jnp.argsort(lambdas)
    lambdas = jnp.take_along_axis(lambdas, idx, -1)
    active = jnp.cumsum((jnp.float32(idx < x.shape[-1])) * 2 - 1, axis=-1)[..., :-1]
    diffs = jnp.diff(lambdas, n=1, axis=-1)
    left = (ones * eps).sum(axis=-1)
    left = left.reshape(*left.shape, 1)
    totals = left + jnp.cumsum(active*diffs, axis=-1)

    def generate_vmap(counter, func):
        if counter == 0:
            return func
        else:
            return generate_vmap(counter - 1, jax.vmap(func))
                
    i = jnp.expand_dims(generate_vmap(len(totals.shape) - 1, partial(jnp.searchsorted, v=1))(totals), -1)
    lam = (1 - jnp.take_along_axis(totals, i, -1)) / jnp.take_along_axis(active, i, -1) + jnp.take_along_axis(lambdas, i+1, -1)
    return jnp.clip(x + lam, eps, 1)