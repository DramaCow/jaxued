import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple

Carry = Any
Output = Any

class ResetRNN(nn.Module):
    """This is a wrapper around an RNN that automatically resets the hidden state upon observing a `done` flag. In this way it is compatible with the jax-style RL loop where episodes automatically end/restart.
    """
    cell: nn.RNNCellBase

    @nn.compact
    def __call__(
        self,
        inputs: Tuple[jax.Array, jax.Array],
        *,
        initial_carry: Optional[Carry] = None,
        reset_carry: Optional[Carry] = None,
    ) -> Tuple[Carry, Output]:
        # On episode completion, model resets to this
        if reset_carry is None:
            reset_carry = self.cell.initialize_carry(jax.random.PRNGKey(0), inputs[0].shape[1:])
        carry = initial_carry if initial_carry is not None else reset_carry

        def scan_fn(cell, carry, inputs):
            x, resets = inputs
            carry = jax.tree_util.tree_map(
                lambda a, b: jnp.where(resets[:, None], a, b), reset_carry, carry
            )
            return cell(carry, x)

        scan = nn.scan(
            scan_fn,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )

        return scan(self.cell, carry, inputs)
