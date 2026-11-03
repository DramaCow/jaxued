"""Exercise the example's real collector, LSTM and PPO replay across batches."""
import importlib.util
from pathlib import Path

import flax.struct
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

RUNNER_PATHS = ['examples/maze_dr.py']


@flax.struct.dataclass
class Observation:
    image: jax.Array
    agent_dir: jax.Array
    has_key: jax.Array


class CounterEnv:
    """Two independent deterministic streams, with different termination times."""
    @staticmethod
    def observe(state):
        step, horizon = state
        return Observation(jnp.ones((5, 5, 3)) * (step + 1) / horizon,
                           jnp.asarray(0, jnp.int32), jnp.asarray(False))

    def step(self, rng, state, action, params):
        step, horizon = state
        done = step + 1 == horizon
        next_state = (jnp.where(done, 0, step + 1), horizon)
        return self.observe(next_state), next_state, done.astype(jnp.float32), done, {}


@pytest.fixture(params=RUNNER_PATHS)
def runner(request):
    path = Path(__file__).resolve().parents[1] / request.param
    spec = importlib.util.spec_from_file_location('boundary_' + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_state(module):
    env = CounterEnv()
    env_state = (jnp.zeros(2, jnp.int32), jnp.array([4, 5]))
    obs = jax.vmap(env.observe)(env_state)
    network = module.ActorCritic(action_dim=3)
    hstate = network.initialize_carry((2,))
    x = jax.tree_util.tree_map(lambda a: a[None], (obs, jnp.zeros(2, bool)))
    params = network.init(jax.random.PRNGKey(1), x, hstate)
    kwargs = dict(apply_fn=network.apply, params=params, tx=optax.adam(1e-3),
                  update_count=0, last_hstate=hstate, last_obs=obs,
                  last_env_state=env_state)
    # Also execute against the unmodified runner to demonstrate regression failure.
    if 'last_done' in module.TrainState.__dataclass_fields__:
        kwargs['last_done'] = jnp.zeros(2, bool)
    return env, module.TrainState.create(**kwargs)


def collect(module, env, key, state, steps):
    return module.sample_trajectories_rnn(key, env, None, state, state.last_hstate,
        state.last_obs, state.last_env_state, 2, steps)


def continue_state(state, final, trajectory):
    fields = dict(last_hstate=final[2], last_obs=final[3], last_env_state=final[4])
    if hasattr(state, 'last_done'):
        fields['last_done'] = trajectory[3][-1]
    return state.replace(**fields)


def assert_tree_close(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x, y, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('split', [3, 4])
def test_split_rollout_preserves_episode_boundaries(runner, split):
    env, state = create_state(runner)
    key = jax.random.PRNGKey(17)
    full, trajectory = jax.jit(lambda k, s: collect(runner, env, k, s, 8))(key, state)
    first, t1 = jax.jit(lambda k, s: collect(runner, env, k, s, split))(key, state)
    np.testing.assert_array_equal(t1[3][-1], [split == 4, False])
    next_state = continue_state(state, first, t1)
    final, t2 = jax.jit(lambda k, s: collect(runner, env, k, s, 8-split))(first[0], next_state)
    assert_tree_close(full[2:], final[2:])
    assert_tree_close(trajectory, jax.tree_util.tree_map(lambda a, b: jnp.concatenate((a, b)), t1, t2))
    np.testing.assert_array_equal(full[0], final[0])


def test_ppo_replay_uses_collected_first_row_reset(runner):
    env, state = create_state(runner)
    first, t1 = collect(runner, env, jax.random.PRNGKey(17), state, 4)
    state = continue_state(state, first, t1)
    # Real nonzero LSTM state and a mixed terminal/nonterminal boundary.
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree_util.tree_leaves(state.last_hstate))
    assert hasattr(state, 'last_done'), 'Complete rollout state must retain terminal flags'
    np.testing.assert_array_equal(state.last_done, [True, False])
    final, t2 = collect(runner, env, first[0], state, 4)
    obs, actions, _, dones, log_probs, values, _ = t2
    # Targets equal collected values, so replay under unchanged parameters has zero
    # critic error. Incorrect first-row reset masks break this identity.
    batch = (obs, actions, dones, log_probs, values, values, jnp.zeros_like(values))
    _, losses = runner.update_actor_critic_rnn(first[0], state, state.last_hstate,
        batch, 2, 4, 1, 1, 0.2, 0.0, 1.0, update_grad=False)
    np.testing.assert_allclose(losses[1][0], 0, atol=1e-9)
    restored = serialization.from_bytes(state, serialization.to_bytes(state))
    np.testing.assert_array_equal(restored.last_done, state.last_done)
    restored_final, restored_trajectory = collect(runner, env, first[0], restored, 4)
    assert_tree_close(final[2:], restored_final[2:])
    assert_tree_close(t2, restored_trajectory)
