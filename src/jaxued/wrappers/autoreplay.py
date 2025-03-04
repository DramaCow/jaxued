import jax
import jax.numpy as jnp
from typing import Any, Tuple, Union
import chex
from flax import struct
from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import EnvState, Observation, Level, EnvParams

@struct.dataclass
class AutoReplayState:
    env_state: EnvState
    level: Level

class AutoReplayWrapper(UnderspecifiedEnv):
    """
    This wrapper replay the **same** level over and over again by resetting to the same level after each episode.
    This is useful for training/rolling out multiple times on the same level.
    """
    
    def __init__(self, env: UnderspecifiedEnv):
        self._env = env
        
    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params
    
    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng_reset, rng_step = jax.random.split(rng)
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, state.level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )
        env_state = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state.replace(env_state=env_state), reward, done, info

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoReplayState(env_state=env_state, level=level)
    
    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)