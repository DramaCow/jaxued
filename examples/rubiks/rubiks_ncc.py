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

import jumanji

from flax import core, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState

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

LAYER_WIDTH = 512
class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class RewardFn(jumanji.environments.logic.rubiks_cube.reward.RewardFn):
    def __call__(self, state) -> chex.Array:
        arr = state.cube
        count_fn = lambda x: (x == x[0, 0]).all()
        return jax.vmap(count_fn)(arr).sum()

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

class LevelSampler(BaseLevelSampler):

    def level_weights(self, sampler, *args,**kwargs):
        return sampler["scores"]
    
    def initialize(self, levels, level_extras):
        sampler = {
                "levels": levels,
                "scores": jnp.full(self.capacity, 1 / self.capacity, dtype=jnp.float32),
                "timestamps": jnp.zeros(self.capacity, dtype=jnp.int32),
                "size": self.capacity,
                "episode_count": 0,
        }
        if level_extras is not None:
            sampler["levels_extra"] = level_extras
        return sampler

@struct.dataclass
class CubeAction:
    face: distrax.Categorical
    depth: distrax.Categorical
    direction: distrax.Categorical

    def sample(self, rng):
        rng1, rng2, rng3 = jax.random.split(rng)
        return jnp.array((
            self.face.sample(rng1),
            self.depth.sample(rng2),
            self.direction.sample(rng3)
        ))

class EnvWrapper:

    def __init__(self, env):
        self.env = env
        self.default_params = None

    def reset_to_level(self, _, level, _2):
        return level.init_timestep.observation, level.init_state

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, rng_unused, state, action, _):
        state, timestep = self.env.step(state, action)
        return timestep.observation, state, timestep.reward, timestep.step_type == 2, timestep.extras

    


@struct.dataclass
class Level:
    init_state: jumanji.env.State
    init_timestep: jumanji.types.TimeStep

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

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

def update_actor_critic(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            obs, actions, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                pi, values_pred = train_state.apply_fn(params, obs)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)

            grad_norm = jnp.linalg.norm(jnp.concatenate(jax.tree_util.tree_map(lambda x: x.flatten(), jax.tree_util.tree_flatten(grads)[0])))
            return train_state, (loss, grad_norm)

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.take(
                x.reshape((-1, *x.shape[2:])),
                jax.random.permutation(rng_perm, num_envs * n_steps),
                axis=0,
            ).reshape((n_minibatch, -1, *x.shape[2:])),
            batch,
        )
        train_state, (losses, grads) = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), (losses, grads)

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

def generate_world(rng, env):

    state, timestep = env.reset(rng)
    return Level(
        init_state = state, 
        init_timestep = timestep
    )

def sample_trajectories_and_learn(env: UnderspecifiedEnv, env_params: EnvParams, config: dict,
                                  rng: chex.PRNGKey, train_state: TrainState, init_obs: Observation, init_env_state: EnvState, update_grad: bool=True) -> Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array, chex.Array, chex.ArrayTree, chex.Array]]:
    """This function loops the following:
        - rollout for config['num_steps']
        - learn / update policy
    
    And it loops it for config['outer_rollout_steps'].
    What is returns is a new carry (rng, train_state, init_obs, init_env_state), and concatenated rollouts. The shape of the rollouts are config['num_steps'] * config['outer_rollout_steps']. In other words, the trajectories returned by this function are the same as if we ran rollouts for config['num_steps'] * config['outer_rollout_steps'] steps, but the agent does perform PPO updates in between.

    Args:
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        config (dict): 
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_obs (Observation): 
        init_env_state (EnvState): 
        update_grad (bool, optional): Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, Observation, EnvState], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict, chex.Array, chex.Array, chex.ArrayTree, chex.Array]]: This returns a tuple:
        (
            (rng, train_state, init_obs, init_env_state),
            (obs, actions, rewards, dones, log_probs, values, info, advantages, targets, losses, grads)
        )
    """
    

    def single_step(carry, _):
        rng, train_state, init_obs, init_env_state = carry
        (
            (rng, train_state, last_obs, last_env_state, last_value),
            (obs, actions, rewards, dones, log_probs, values, info),
        ) = sample_trajectories(
            rng,
            env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            config["num_train_envs"],
            config["num_steps"],
        )
        advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
        
        # Update the policy using trajectories collected from replay levels
        (rng, train_state), (losses, grads) = update_actor_critic(
            rng,
            train_state,
            (obs, actions, log_probs, values, targets, advantages),
            config["num_train_envs"],
            config["num_steps"],
            config["num_minibatches"],
            config["epoch_ppo"],
            config["clip_eps"],
            config["entropy_coeff"],
            config["critic_coeff"],
            update_grad=update_grad,
        )
        new_carry = (rng, train_state, last_obs, last_env_state)
        return new_carry, (obs, actions, rewards, dones, log_probs, values, info, advantages, targets, losses, grads)

    
    carry = (rng, train_state, init_obs, init_env_state)
    new_carry, all_rollouts = jax.lax.scan(single_step, carry, None, length=config['outer_rollout_steps'])

    all_rollouts = jax.tree_util.tree_map(lambda x: jnp.concatenate(x, axis=0), all_rollouts)
    return new_carry, all_rollouts

def evaluate(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
    keep_states=True
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the model on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        pi, _ = train_state.apply_fn(train_state.params, obs)
        action = pi.sample(seed=rng_action)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        if keep_states:
            return (rng, obs, next_state, done, next_mask, episode_length), (state, reward)
        else:
            return (rng, obs, next_state, done, next_mask, episode_length), (None, reward)
    
    (_, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths

class ActorCritic(nn.Module):
    """Non-recurrent network from here: https://github.com/MichaelTMatthews/Craftax"""
    cube_dim: int
    
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

        face = nn.Dense(
            6, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        depth = nn.Dense(
            self.cube_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        direction = nn.Dense(
            3, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi1 = distrax.Categorical(logits=face)
        pi2 = distrax.Categorical(logits=depth)
        pi3 = distrax.Categorical(logits=direction)
        pi = CubeAction(
            face = pi1,
            depth = pi2,
            direction = pi3
        )

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
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)

    dist = train_state.sampler["scores"]
    return {
        "log":{
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
            # "level_sampler/adv_entropy": -jnp.log(dist + 1e-6).T @ dist,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }

def compute_score(config: dict, dones: chex.Array, values: chex.Array, max_returns: chex.Array, advantages: chex.Array) -> chex.Array:
    # Computes the score for each level
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def main(config=None, project="JAXUED_TEST"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]
        env_steps_delta = config["eval_freq"] * config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps_delta / stats['time_delta'],
        }
        
        # evaluation performance
        returns     = stats["eval_returns"]
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        losses = stats["losses"]

        log_dict.update({
            "total_loss": losses[0][-1],
            "value_loss": losses[1][0][-1],
            "policy_loss": losses[1][1][-1],
            "entropy_loss": losses[1][2][-1],
            "adv_loss": metrics["adv_loss"][-1],
            "adv_entropy": metrics["adv_entropy"][-1]
        })

        # level sampler
        log_dict.update(train_state_info["log"])

        # images
        # log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
        # log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

        # for s in ['dr', 'replay', 'mutation']:
        #     if train_state_info['info'][f'num_{s}_updates'] > 0:
        #         log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

        # i = 0
        # frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
        # frames = np.array(frames[:episode_length])
        # log_dict.update({f"animations/animation": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)
    
    env = EnvWrapper(jumanji.make("RubiksCube-v0", reward_fn=RewardFn()))
    eval_env = env
    env_params = env.default_params

    def sample_random_level(rng):
        return env.reset(rng)
    
    # Setup the environment. 
    # TODO: Add support for Pixels
        
    # And the level sampler    
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config['topk_k']},
        duplicate_check=config['buffer_duplicate_check'],
    )

    @partial(jax.jit, static_argnums=(2, ))
    def learnability_fn(rng, levels, num_envs, train_state):
        def rollout_fn(rng):

            # Get the scores of the levels
            rng, _rng = jax.random.split(rng)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, num_envs), levels, env_params)
            # Rollout
            (
                (rng, _, last_obs, last_state, last_value, disc_return),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                num_envs,
                500,
                1.0,
                True
            )
            return disc_return
        
        rng, _rng = jax.random.split(rng)
        num_samples = 5
        returns = jax.vmap(rollout_fn)(jax.random.split(_rng, num_samples))
        
        # fit gaussian to mean episode returns
        
        def gaussian_pdf(x, mean, var):
            return (1 / (jnp.sqrt(2 * jnp.pi) * var)) * jnp.exp(-0.5 * ((x - mean) / var) ** 2)
        
        mean_returns = jnp.mean(returns, axis=0)
        mu = mean_returns.mean()
        sigma_2 = mean_returns.var()
        
        pdf_values = gaussian_pdf(mean_returns, mu, sigma_2)
        
        scores = jnp.sqrt(returns.var(axis=0)) / jnp.sqrt(num_samples) * pdf_values
        
        return scores, returns.max(axis=0)

    def replace_fn(rng, train_state, old_level_scores):
        # NOTE: scores here are the actual UED scores, NOT the probabilities induced by the projection

        # Sample new levels
        rng, _rng = jax.random.split(rng)
        new_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["num_train_envs"]))

        rng, _rng = jax.random.split(rng)
        new_level_scores, max_returns = learnability_fn(_rng, new_levels, config["num_train_envs"], train_state)

        idxs = jnp.flipud(jnp.argsort(new_level_scores))

        new_levels = jax.tree_util.tree_map(
            lambda x: x[idxs], new_levels
        )
        new_level_scores = new_level_scores[idxs]

        update_sampler = {**train_state.sampler,"scores": old_level_scores}

        sampler, _ = level_sampler.insert_batch(update_sampler, new_levels, new_level_scores, {"max_return": max_returns})
        
        return sampler
    
    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / (config["num_updates"] * config['outer_rollout_steps'])
            )
            return config["lr"] * frac
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )

        network = ActorCritic(config["cube_dim"])
        network_params = network.init(rng, obs)
        tx = optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                ti_ada(vy0 = jnp.zeros(config["level_buffer_capacity"]), eta=linear_schedule),
                # optax.scale_by_optimistic_gradient()
            )
        rng, _rng = jax.random.split(rng)
        init_levels = jax.vmap(sample_random_level)(jax.random.split(_rng, config["level_buffer_capacity"]))
        sampler = level_sampler.initialize(init_levels, {"max_return": jnp.full(config["level_buffer_capacity"], -jnp.inf)})
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=init_levels,
            replay_last_level_batch=init_levels,
            mutation_last_level_batch=init_levels,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        
        rng, train_state, xhat, prev_grad, y_opt_state = carry

        new_score = projection_simplex_truncated(xhat + prev_grad, config["meta_trunc"]) # if config["META_OPTIMISTIC"] else xhat
        sampler = {**train_state.sampler, "scores": new_score}
        # Collect trajectories on replay levels
        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])

        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
        (
            (rng, train_state, last_obs, last_env_state),
            (obs, actions, rewards, dones, log_probs, values, info, advantages, targets, losses, grads)
            ) = sample_trajectories_and_learn(env, env_params, config,
                                rng, train_state, init_obs, init_env_state, update_grad=True)
        jax.debug.print("{}",  (info["returned_episode_returns"] * dones).sum() / dones.sum())
        
        # Update the level sampler
        levels = sampler["levels"]
        rng, _rng = jax.random.split(rng)
        scores, _ = learnability_fn(_rng, levels, config['level_buffer_capacity'], train_state)

        # jax.debug.print("top 10 scores: {}", jax.lax.top_k(scores, 10))

        rng, _rng = jax.random.split(rng)
        new_sampler = replace_fn(_rng, train_state, scores)
        sampler = {**new_sampler, "scores": new_score}

        grad_fn = jax.grad(lambda y: y.T @ new_sampler["scores"] - 0.001 * jnp.log(y + 1e-6).T @ y)

        grad, y_opt_state = y_ti_ada.update(grad_fn(new_score), y_opt_state)
        xhat = projection_simplex_truncated(xhat + grad, config["meta_trunc"])

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "achievements": (info["achievements"] * dones[..., None]).sum(axis=0).sum(axis=0) / dones.sum(),
            "achievement_count": (info["achievement_count"] * dones).sum() / dones.sum(),
            "returned_episode_lengths": (info["returned_episode_lengths"] * dones).sum() / dones.sum(),
            "max_episode_length": info["returned_episode_lengths"].max(),
            "levels_played": init_env_state.env_state,
            "mean_returns": (info["returned_episode_returns"] * dones).sum() / dones.sum(),
            "grad_norms": grads.mean(),
            "adv_loss": (lambda y: y.T @ new_sampler["scores"] - 0.01 * jnp.log(y + 1e-6).T @ y)(new_score),
            "adv_entropy": -jnp.log(new_score + 1e-6).T @ new_score
        }

        train_state = train_state.replace(
            opt_state = jax.tree_util.tree_map(
                lambda x: x if type(x) is not ScaleByTiAdaState else x.replace(vy = y_opt_state.vy), train_state.opt_state
            ),
            sampler = sampler,
            update_state=UpdateState.REPLAY,
            num_replay_updates=train_state.num_replay_updates + 1,
            replay_last_level_batch=levels,
        )
        
        return (rng, train_state, xhat, prev_grad, y_opt_state), metrics
    
    
    def eval(rng: chex.PRNGKey, train_state: TrainState, keep_states=True):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        num_levels = config['n_eval_levels']
        levels = jax.vmap(generate_world, (0, None, None))(jax.random.split(jax.random.PRNGKey(101), num_levels), env_params, DEFAULT_STATICS)
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate(
            rng,
            eval_env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            config['num_eval_steps'], keep_states=keep_states
        )
        mask = jnp.arange(config['num_eval_steps'])[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state, xhat, prev_grad, y_opt_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state, False)
        
        # Collect Metrics
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        # And one attempt
        # states = jax.tree_util.tree_map(lambda x: x[:, :1], states)
        episode_lengths = episode_lengths[:1]
        # images = jax.vmap(jax.vmap(render_craftax_pixels, (0, None)), (0, None))(states.env_state.env_state, BLOCK_PIXEL_SIZE_IMG) # (num_steps, num_eval_levels, ...)
        # frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_ep_lengths"]  = episode_lengths
        # metrics["eval_animation"] = (frames, episode_lengths)
        
        max_num_images = 32

        metrics["dr_levels"] = None # jax.vmap(render_craftax_pixels, (0, None))(jax.tree_util.tree_map(lambda x: x[:max_num_images], train_state.dr_last_level_batch), BLOCK_PIXEL_SIZE_IMG)
        metrics["replay_levels"] = None # jax.vmap(render_craftax_pixels, (0, None))(jax.tree_util.tree_map(lambda x: x[:max_num_images], train_state.replay_last_level_batch), BLOCK_PIXEL_SIZE_IMG)
        metrics["mutation_levels"] = None # jax.vmap(render_craftax_pixels, (0, None))(jax.tree_util.tree_map(lambda x: x[:max_num_images], train_state.mutation_last_level_batch), BLOCK_PIXEL_SIZE_IMG)
        
        # highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
        # highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())
        
        metrics["highest_scoring_level"] = None # render_craftax_pixels(highest_scoring_level, BLOCK_PIXEL_SIZE_IMG)
        metrics["highest_weighted_level"] = None # render_craftax_pixels(highest_weighted_level, BLOCK_PIXEL_SIZE_IMG)
        
        return (rng, train_state, xhat, prev_grad, y_opt_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state, False)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths))
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval':
        return eval_checkpoint(config, ) # evaluate and exit early

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)

    # Set up y optimizer state
    y_ti_ada = scale_y_by_ti_ada(eta=config["meta_lr"])
    y_opt_state = y_ti_ada.init(jnp.zeros_like(train_state.sampler["scores"]))
        
    xhat = grad = jnp.zeros_like(train_state.sampler["scores"])
    runner_state = (rng_train, train_state, xhat, grad, y_opt_state)

    runner_state = (rng, train_state, xhat, grad, y_opt_state)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    return runner_state[1]

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="nate_jaxued")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    # === Train vs Eval ===
    parser.add_argument("--env_name", type=str, choices=['Craftax-Symbolic-v1', 'Craftax-Pixels-v1', 'Craftax-Classic-Symbolic-v1', 'Craftax-Classic-Pixels-v1'], default='Craftax-Symbolic-v1')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=0)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--eval_num_attempts", type=int, default=1)
    group = parser.add_argument_group('Training params')
    # === PPO === 
    group.add_argument("--lr", type=float, default=2e-4)
    group.add_argument("--max_grad_norm", type=float, default=1.0)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=250)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--outer_rollout_steps", type=int, default=1)
    group.add_argument("--num_train_envs", type=int, default=1024)
    group.add_argument("--num_minibatches", type=int, default=8)
    group.add_argument("--gamma", type=float, default=0.99)
    group.add_argument("--epoch_ppo", type=int, default=4)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.8)
    group.add_argument("--entropy_coeff", type=float, default=0.01)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    group.add_argument("--meta_lr", type=float, default=1e-2)
    group.add_argument("--meta_trunc", type=float, default=1e-5)
    # === PLR ===
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.5)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    # === ACCEL ===
    parser.add_argument("--use_accel",                          action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_edits",                          type=int, default=30)
    parser.add_argument("--accel_mutation",                     type=str, default="swap", choices=["swap", "swap_restricted", "noise"])

    # === Eval CONFIG ===
    parser.add_argument("--n_eval_levels", type=int, default=5)
    parser.add_argument("--num_eval_steps", type=int, default=2048)
    # === DR CONFIG ===
    
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"] * config["outer_rollout_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    wandb.login()
    main(config, project=config["project"])
