import json
import os
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jaxued.environments.underspecified_env import EnvParams, EnvState, UnderspecifiedEnv
from jaxued.utils import compute_max_mean_returns_epcount
import optax
from flax import struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.maze.env_editor import MazeEditor, Observation
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level
from jaxued.wrappers import AutoReplayWrapper
import chex

# region PPO helper functions    
@struct.dataclass
class TrainState:
    update_count: int
    pro_train_state: BaseTrainState
    ant_train_state: BaseTrainState
    adv_train_state: BaseTrainState

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

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:
        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

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
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
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

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
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
        init_hstate (chex.ArrayTree): 
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
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
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
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
    
class AdversaryActorCritic(nn.Module):
    # The adversary's network architecture
    action_dim: Sequence[int]
    max_timesteps: int = 50
    
    @nn.compact
    def __call__(self, inputs: Tuple[Observation, chex.Array], hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(128, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        time_value = nn.Embed(self.max_timesteps + 1, 10, name="time_embed", embedding_init=orthogonal(1.0))(jnp.clip(obs.time, None, self.max_timesteps))
        random_z_value = obs.random_z
        embedding = jnp.concatenate((img_embed, time_value, random_z_value), axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)

        # Mask out this
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
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

def main(config=None, project="JAXUED_TEST"):
    run = wandb.init(config=config, project=project, group=config["group_name"], tags=["PAIRED",])
    config = wandb.config
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("misc/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_length/*", step_metric="num_updates")

    def log_eval(stats):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = 2 * stats["update_count"] * config["num_train_envs"] * config["student_num_steps"]
        log_dict = {
            "misc/mean_num_blocks": stats["mean_num_blocks"].mean(),
            "misc/num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "misc/sps": env_steps / stats['time_delta'],
            "misc/prot_perf_mean": stats['pro_mean_returns'].mean(),
            "misc/ant_perf_max":   stats['ant_max_returns'].mean(),
            "misc/prot_perf_max":  stats['pro_max_returns'].mean(),
            "misc/ant_perf_mean":  stats['ant_mean_returns'].mean(),
            "misc/ant_num_episodes": stats['ant_eps'].mean(),
            "misc/pro_num_episodes": stats['pro_eps'].mean(),
            "misc/regret":   stats['est_regret'].mean(),
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})
        def make_caption(i):
            pro_mean_returns = jnp.round(stats['pro_mean_returns'][-1][i], 2) # .flatten()
            ant_max_returns  = jnp.round(stats['ant_max_returns'][-1][i], 2) # .flatten()
            est_regret       = jnp.round(stats['est_regret'][-1][i], 2) # .flatten()
            return f"P({pro_mean_returns:.2f})|A({ant_max_returns:.2f})|R({est_regret:.2f})"
    
        log_dict.update({f"images/levels": [wandb.Image(np.array(image), caption=make_caption(i)) for i, image in enumerate(stats["levels"])]})

        # animations
        for i, level_name in enumerate(config["eval_levels"]):
            frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
            frames = np.array(frames[:episode_length])
            log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)
    
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    adv_env = MazeEditor(env, random_z_dimensions=config['adv_random_z_dimension'], zero_out_random_z=config['adv_zero_out_random_z'])
    eval_env = env
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    adv_env_params = adv_env.default_params

    def sample_empty_level():
        w, h = env._env.max_width, env._env.max_height
        return Level(
            wall_map=jnp.zeros((h, w), dtype=jnp.bool_),
            width=w,
            height=h,
            
            # These values don't matter, as the adversary overwrites them.
            goal_pos=jnp.array([0, 0], dtype=jnp.uint32),
            agent_pos=jnp.array([1, 1], dtype=jnp.uint32),
            agent_dir=jnp.array(0, dtype=jnp.uint8),
        )

    @jax.jit
    def create_train_state(rng):
        def create_inner_train_state(rng, env, env_params, network_cls, prefix, network_kws={}):
            def linear_schedule(count):
                frac = (
                    1.0
                    - (count // (config[f"{prefix}num_minibatches"] * config[f"{prefix}epoch_ppo"]))
                    / config["num_updates"]
                )
                return config[f"{prefix}lr"] * frac
            obs, _ = env.reset_to_level(rng, sample_empty_level(), env_params)
            obs = jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
                obs,
            )
            init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
            network = network_cls(env.action_space(env_params).n, **network_kws)
            network_params = network.init(rng, init_x, network_cls.initialize_carry((config["num_train_envs"],)))
            tx = optax.chain(
                optax.clip_by_global_norm(config[f"{prefix}max_grad_norm"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
                # optax.adam(learning_rate=config[f"{prefix}lr"], eps=1e-5),
            )
            return BaseTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        rng_pro, rng_ant, rng_adv = jax.random.split(rng, 3)
        return TrainState(
            update_count = 0,
            pro_train_state = create_inner_train_state(rng_pro, env, env_params, ActorCritic, "student_"),
            ant_train_state = create_inner_train_state(rng_ant, env, env_params, ActorCritic, "student_"),
            adv_train_state = create_inner_train_state(rng_adv, adv_env, adv_env_params, AdversaryActorCritic, "adv_", network_kws={"max_timesteps": config["adv_num_steps"]})
        )

    def train_step(carry, _):
        def rollout(rng, env, env_params, train_state, init_hstate, levels, num_steps, prefix):
            # Single rollout
            rng, _rng = jax.random.split(rng)
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(_rng, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                config["num_train_envs"],
                num_steps,
            )
            advantages, targets = compute_gae(config[f"{prefix}gamma"], config[f"{prefix}gae_lambda"], last_value, values, rewards, dones)
            return (obs, actions, dones, log_probs, values, targets, advantages), (dones, rewards, last_env_state, last_value, info)
        
        def update(rng, train_state, init_hstate, rollout, prefix):
            # Returns: (rng, train_state), losses
            return update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate,
                rollout,
                config["num_train_envs"],
                config[f"{prefix}num_steps"],
                config[f"{prefix}num_minibatches"],
                config[f"{prefix}epoch_ppo"],
                config[f"{prefix}clip_eps"],
                config[f"{prefix}entropy_coeff"],
                config[f"{prefix}critic_coeff"],
                update_grad=True,
            )
        
        rng, train_state = carry
        
        pro_train_state = train_state.pro_train_state
        ant_train_state = train_state.ant_train_state
        adv_train_state = train_state.adv_train_state
        
        # adversary rollout (aka level generation)
        rng, _rng = jax.random.split(rng)
        empty_levels = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), sample_empty_level())
        adv_rollout, (_, _, last_env_state, last_value, _) = rollout(_rng, adv_env, adv_env_params, adv_train_state, AdversaryActorCritic.initialize_carry((config["num_train_envs"],)), empty_levels, config["adv_num_steps"], "adv_")
        levels = last_env_state.level

        # protagonist rollout
        rng, _rng = jax.random.split(rng)
        pro_rollout, (dones, rewards, _, _, _) = rollout(_rng, env, env_params, pro_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), levels, config["student_num_steps"], "student_")
        pro_mean_returns, pro_max_returns, pro_eps = compute_max_mean_returns_epcount(dones, rewards)

        # antagonist rollout
        rng, _rng = jax.random.split(rng)
        ant_rollout, (dones, rewards, _,  _, _) = rollout(_rng, env, env_params, ant_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), levels, config["student_num_steps"], "student_")
        ant_mean_returns, ant_max_returns, ant_eps = compute_max_mean_returns_epcount(dones, rewards)

        # Adversary Rewards
        est_regret = ant_max_returns - pro_mean_returns
        obs, actions, dones, log_probs, values, targets, advantages = adv_rollout
        rewards = jnp.zeros_like(values).at[-1].set(est_regret)
        advantages, targets = compute_gae(config["adv_gamma"], config["adv_gae_lambda"], last_value, values, rewards, dones)
        adv_rollout = (obs, actions, dones.at[-1].set(True), log_probs, values, targets, advantages)
                
        (rng, pro_train_state), pro_losses = update(rng, pro_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), pro_rollout, "student_")
        (rng, ant_train_state), ant_losses = update(rng, ant_train_state, ActorCritic.initialize_carry((config["num_train_envs"],)), ant_rollout, "student_")
        (rng, adv_train_state), adv_losses = update(rng, adv_train_state, AdversaryActorCritic.initialize_carry((config["num_train_envs"],)), adv_rollout, "adv_")
        
        metrics = {
            "pro_losses": jax.tree_util.tree_map(lambda x: x.mean(), pro_losses),
            "ant_losses": jax.tree_util.tree_map(lambda x: x.mean(), ant_losses),
            "adv_losses": jax.tree_util.tree_map(lambda x: x.mean(), adv_losses),
            "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
            "pro_mean_returns": pro_mean_returns,
            "ant_max_returns":  ant_max_returns,
            "pro_max_returns":  pro_max_returns,
            "ant_mean_returns": ant_mean_returns,
            "est_regret":       est_regret,
            "pro_eps": pro_eps,
            "ant_eps": ant_eps,
            "levels": levels,
        }
        
        train_state = train_state.replace(
            update_count=train_state.update_count + 1,
            pro_train_state=pro_train_state,
            ant_train_state=ant_train_state,
            adv_train_state=adv_train_state,
        )
        return (rng, train_state), metrics
    
    def eval(rng, train_state):
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])
        
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state.pro_train_state)
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
        frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
        
        metrics["update_count"] = train_state.update_count
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["eval_animation"] = (frames, episode_lengths)
        metrics["levels"] = jax.vmap(env_renderer.render_level, (0, None))(jax.tree_util.tree_map(lambda x: x[-1], metrics["levels"]), env_params)
        
        return (rng, train_state), metrics
    
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
            params = loaded_checkpoint['pro_train_state']['params']
            train_state = train_state_og.replace(pro_train_state=train_state_og.pro_train_state.replace(params=params))
            return train_state.pro_train_state, config
        
        train_state, config = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['eval_levels'])
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval': return eval_checkpoint(config) # evaluate and exit early

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)
    
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    start_time = time.time()
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    return runner_state[1]

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--eval_levels", nargs='+', default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group('Training params')
    # === PPO === ^--(.*)".*  ||  $1:
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_train_envs", type=int, default=32)
    # === Student hyperparams ===
    group.add_argument("--student_num_steps", type=int, default=256)
    group.add_argument("--student_lr", type=float, default=1e-4)
    group.add_argument("--student_max_grad_norm", type=float, default=0.5)
    group.add_argument("--student_num_minibatches", type=int, default=1)
    group.add_argument("--student_gamma", type=float, default=0.995)
    group.add_argument("--student_epoch_ppo", type=int, default=5)
    group.add_argument("--student_clip_eps", type=float, default=0.2)
    group.add_argument("--student_gae_lambda", type=float, default=0.98)
    group.add_argument("--student_entropy_coeff", type=float, default=1e-3)
    group.add_argument("--student_critic_coeff", type=float, default=0.5)
    # === Adversary hyperparams ===
    group.add_argument("--adv_random_z_dimension", type=int, default=16)
    group.add_argument("--adv_zero_out_random_z", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--adv_num_steps", type=int, default=50)
    group.add_argument("--adv_lr", type=float, default=1e-4)
    group.add_argument("--adv_max_grad_norm", type=float, default=0.5)
    group.add_argument("--adv_num_minibatches", type=int, default=1)
    group.add_argument("--adv_gamma", type=float, default=0.995)
    group.add_argument("--adv_epoch_ppo", type=int, default=5)
    group.add_argument("--adv_clip_eps", type=float, default=0.2)
    group.add_argument("--adv_gae_lambda", type=float, default=0.98)
    group.add_argument("--adv_entropy_coeff", type=float, default=1e-3)
    group.add_argument("--adv_critic_coeff", type=float, default=0.5)
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (2 * config["num_train_envs"] * config["student_num_steps"])
    config["group_name"] = 'paired4_' + ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
