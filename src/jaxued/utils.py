import jax
import jax.numpy as jnp

# This file is a modified version of
# https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/util/rl/ued_scores.py.
# Credit: minimax

def accumulate_rollout_stats(dones, metrics, *, time_average):
    def iter(carry, input):
        sum_val, max_val, accum_val, step_count, episode_count = carry
        done, step_val = input
        
        accum_val = jax.tree_util.tree_map(lambda x, y: x + y, accum_val, step_val)
        step_count += 1
        
        if time_average:
            # val = jax.tree_util.tree_map(lambda x, b: jax.lax.select(b, x / step_count, x), accum_val, time_average)
            val = jax.tree_util.tree_map(lambda x: x / step_count, accum_val)
        else:
            val = accum_val
        
        sum_val = jax.tree_util.tree_map(lambda x, y: x + done * y, sum_val, val)
        max_val = jax.tree_util.tree_map(lambda x, y: (1 - done) * x + done * jnp.maximum(x, y), max_val, val)
        
        episode_count += done
        
        accum_val = jax.tree_util.tree_map(lambda x: (1 - done) * x, accum_val)
        step_count = (1 - done) * step_count
        
        return (sum_val, max_val, accum_val, step_count, episode_count), None
    
    batch_size = dones.shape[1]
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x[0]), metrics)
    (sum_val, max_val, _, _, episode_count), _ = jax.lax.scan(
        iter,
        (zeros, zeros, zeros, jnp.zeros(batch_size, dtype=jnp.uint32), jnp.zeros(batch_size, dtype=jnp.uint32)),
        (dones, metrics),
    )
    
    mean_val = jax.tree_util.tree_map(lambda x: x / jnp.maximum(episode_count, 1), sum_val)
    
    return mean_val, max_val, episode_count

def compute_max_returns(dones, rewards):
    _, max_returns, _ = accumulate_rollout_stats(dones, rewards, time_average=False)
    return max_returns

def compute_max_mean_returns_epcount(dones, rewards):
    return accumulate_rollout_stats(dones, rewards, time_average=False)

def max_mc(dones, values, max_returns, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, max_returns[None, :] - values, time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)

def positive_value_loss(dones, advantages, incomplete_value=-jnp.inf):
    mean_scores, _, episode_count = accumulate_rollout_stats(dones, jnp.maximum(advantages, 0), time_average=True)
    return jnp.where(episode_count > 0, mean_scores, incomplete_value)