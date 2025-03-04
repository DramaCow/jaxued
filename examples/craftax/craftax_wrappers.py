from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from craftax.craftax.constants import Achievement
from flax import struct

from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import (EnvParams, EnvState, Level,
                                                    Observation)


def compute_score(state: EnvState, done: bool):
    achievements = state.achievements * done * 100
    info = {}
    for achievement in Achievement:
        name = f"Achievements/{achievement.name.lower()}"
        info[name] = achievements[achievement.value]
    return jnp.exp(jnp.mean(jnp.log(1 + achievements))) - 1.0


@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(UnderspecifiedEnv):
    """Log the episode returns, lengths and achievements."""

    def __init__(self, env):
        self._env = env
    
    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params
    

    @partial(jax.jit, static_argnums=(0, 3))
    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = LogEnvState(level, 0.0, 0, 0.0, 0, 0)
        return self.get_obs(state), state
    
    @partial(jax.jit, static_argnums=(0, 4))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"]    = state.returned_episode_returns
        info["returned_episode_lengths"]    = state.returned_episode_lengths
        info["timestep"]                    = state.timestep
        info["returned_episode"]            = done
        info['achievements']                = env_state.achievements
        info['achievement_count']           = env_state.achievements.sum()
        
        if hasattr(env_state, 'player_level'):
            info['floor']                       = env_state.player_level
        return obs, state, reward, done, info

    def get_obs(self, state: LogEnvState) -> chex.Array:
        return self._env.get_obs(state.env_state)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)

class CraftaxLoggerGymnaxWrapper(UnderspecifiedEnv):
    """
    Defines interface for environments that are compatible with UED methods.
    Extends Gymnax interface with support for different levels and also uses the logwrapper.
    """
    
    def __init__(self, env):
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
        # step_env used to avoid unnecessary auto-resets
        return self._env.step_env(rng, state, action, params)

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        state = LogEnvState(level, 0.0, 0, 0.0, 0, 0)
        return self._env.get_obs(state), state
    
    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)
