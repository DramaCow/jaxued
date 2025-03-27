import json
import os
import time
from enum import IntEnum
from typing import Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG
from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from craftax.craftax.renderer import render_craftax_pixels as render_pixels
from craftax.craftax_classic.renderer import render_craftax_pixels as render_pixels_classic
from craftax.craftax.world_gen.world_gen import generate_world as generate_world_craftax
from craftax.craftax_classic.world_gen import generate_world as generate_world_classic
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
from flax import core, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
LAYER_WIDTH = 512
from ncc_utils import scale_y_by_ti_ada, ScaleByTiAdaState, ti_ada, projection_simplex_truncated

import wandb
from jaxued.environments.underspecified_env import (EnvParams, EnvState,
                                                    Observation,
                                                    UnderspecifiedEnv)
from jaxued.level_sampler import LevelSampler as BaseLevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

# Hack to resolve paths
from functools import partial
import sys
sys.path.append('.')
from examples.craftax.craftax_wrappers import CraftaxLoggerGymnaxWrapper, LogWrapper
from examples.craftax.mutators import (make_mutator_craftax_mutate_angles,
                       make_mutator_craftax_swap,
                       make_mutator_craftax_swap_restricted)

class ActorCritic(nn.Module):
    """Non-recurrent network from here: https://github.com/MichaelTMatthews/Craftax"""
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        activation = nn.tanh

        actor_mean = nn.Dense(
            LAYER_WIDTH, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            LAYER_WIDTH, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            LAYER_WIDTH, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            LAYER_WIDTH, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            LAYER_WIDTH, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            LAYER_WIDTH, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


NUM_ENVS = 1000
NUM_ATTEMPTS = 10

def sample_trajectories(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    gamma: float = 0.99,
    give_returns: bool = False
) -> Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, obs, env_state, disc_factor, returns, valid_mask = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        pi, value = train_state.apply_fn(train_state.params, obs)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        valid_mask *= ~done
        returns += disc_factor * reward * valid_mask
        disc_factor *= gamma

        carry = (rng, train_state, next_obs, env_state, disc_factor, returns, valid_mask)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, last_obs, last_state, _, returns, _), (obs, action, reward, done, log_prob, value, info) = jax.lax.scan(
        sample_step,
        (rng, train_state, init_obs, init_env_state, 1.0, jnp.zeros(num_envs), jnp.ones(num_envs)),
        None,
        length=max_episode_length,
    )
    
    _, last_value = train_state.apply_fn(train_state.params, last_obs)
    
    if not give_returns:
        return (rng, train_state, last_obs, last_state, last_value), (obs, action, reward, done, log_prob, value, info)
    else:
        return (rng, train_state, last_obs, last_state, last_value, returns), (obs, action, reward, done, log_prob, value, info)

DEFAULT_STATICS = CraftaxSymbolicEnv.default_static_params()
default_env = CraftaxSymbolicEnv(DEFAULT_STATICS)
env = LogWrapper(default_env)
env = AutoReplayWrapper(env)
eval_env = env
env_params = env.default_params
level_rng = jax.random.split(jax.random.key(100), NUM_ENVS)
levels = jax.vmap(generate_world_craftax, in_axes=(0, None, None))(level_rng, env.default_params, DEFAULT_STATICS)

sample_random_level = lambda r: generate_world_craftax(r, env.default_params, DEFAULT_STATICS)
def get_eval_data(params_path):

    rng = jax.random.key(1)

    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], 256, axis=0)[None, ...], 256, axis=0),
        obs,
    )

    network = ActorCritic(env.action_space(env_params).n)

    params = jnp.load(params_path, allow_pickle=True).item()

    train_state = TrainState.create(
        params = params,
        apply_fn=network.apply,
        tx = optax.scale(1.0)
    )

    def rollout_fn(rng, _):

        rng, _rng = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, NUM_ENVS), levels, env_params)
        
        rng, _rng = jax.random.split(rng)
        return rng, sample_trajectories(
            _rng,
            env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            NUM_ENVS,
            500,
            1.0,
            True
        )[0][-1]

    return jax.lax.scan(rollout_fn, rng, None, length=NUM_ATTEMPTS)[1].mean(axis=0)


ALGS = [
    "sfl",
    "plr",
    "ncc",
    "ncc_regret_pvl",
    "dr"
]

if __name__ == "__main__":
    import pandas as pd

    for alg in ALGS:

        for seed in range(5):
            params_path = f"craftax_params/{alg}_params_{seed}.npy"
            returns = get_eval_data(params_path)

            PATH_TO_SAVE = f"craftax_results_1/{alg}/{seed}.csv"
    
            pd.DataFrame({
                'env-id': jnp.arange(len(returns)),
                'returns': returns,
            }).to_csv(PATH_TO_SAVE, index=False)

            print("saved at path", PATH_TO_SAVE)


    