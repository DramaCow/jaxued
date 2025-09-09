import jax
import jax.numpy as jnp
from typing import Any, Tuple, Union, Callable
import chex
from flax import struct
from jaxued.environments import UnderspecifiedEnv
from jaxued.environments.underspecified_env import EnvState, Observation, Level, EnvParams

@struct.dataclass
class AutoResetState:
    env_state: EnvState
    rng: chex.PRNGKey
    
class AutoResetWrapper(UnderspecifiedEnv):
    """
    This is a wrapper around an `UnderspecifiedEnv`, allowing for the environment to be automatically reset upon completion of an episode. This behaviour is similar to the default Gymnax interface. The user can specify a callable `sample_level` that takes in a PRNGKey and returns a level.

    Warning: 
        To maintain compliance with UnderspecifiedEnv interface, user can reset to an
        arbitrary level. This includes levels outside the support of sample_level(). Consequently,
        the tagged rng is defaulted to jax.random.PRNGKey(0). If your code relies on this, careful
        attention may be required.
    """
    
    def __init__(self, env: UnderspecifiedEnv, sample_level: Callable[[chex.PRNGKey], Level]):
        self._env = env
        self.sample_level = sample_level
        
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
        rng_sample, rng_reset, rng_step = jax.random.split(rng, 3)
        
        new_level = self.sample_level(rng_sample)
        
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, new_level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng, state.env_state, action, params
        )
        
        env_state = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        level_rng = jax.lax.select(done, rng_sample, state.rng)
        
        info["rng"] = level_rng
        
        return obs, AutoResetState(env_state=env_state, rng=level_rng), reward, done, info

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoResetState(env_state=env_state, rng=jax.random.PRNGKey(0))
    
    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)
        

@struct.dataclass
class AutoResetFiniteState:
    env_state: EnvState
    level_idx: int

class AutoResetFiniteWrapper(UnderspecifiedEnv):
    """
    On episode termination, the environment is reset to a level specified by some
    fixed, discrete distribution with finite support.
    
    Args:
        env (UnderspecifiedEnv): 
        levels (_type_): The set of levels to sample from.
        p (_type_, optional): probabilities for each level, defaults to uniform. Defaults to None.
    
    Warning: 
        To maintain compliance with the UnderspecifiedEnv interface, one
        **can** still reset to an arbitrary level. This includes levels outside
        the specified distribution. An optional flag is provided such that if the
        user wishes to check if a level is in distribution, the wrapped state will
        be marked with the first index index corresponding to the reset level in
        distribution.
    """
    
    def __init__(self, env: UnderspecifiedEnv, levels, p=None, check_reset_to_level=True):
        self._env = env
        self._num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
        self._levels = levels
        self._p = p if p is not None else (1 / self._num_levels) * jnp.ones(self._num_levels)
        self._check_reset_to_level = True
        
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
        rng_sample, rng_reset, rng_step = jax.random.split(rng, 3)
        
        new_level_idx = jax.random.choice(rng_sample, self._num_levels, p=self._p)
        new_level = jax.tree_util.tree_map(lambda x: x[level_idx], self._levels)
        
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, new_level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng, state.env_state, action, params
        )
        
        env_state = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st)
        obs = jax.tree_util.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        level_idx = jax.lax.select(done, new_level_idx, state.level_idx)
        
        info["level_idx"] = level_idx
        
        return obs, AutoResetFiniteState(env_state=env_state, level_idx=level_idx), reward, done, info

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: Level,
        params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        
        if self._check_reset_to_level:
            eq_tree = jax.tree_util.tree_map(lambda X, y: (X == y).reshape(self._num_levels, -1).all(axis=-1), self._levels, level)
            eq_tree_flat, _ = jax.tree_util.tree_flatten(eq_tree)
            eq_mask = jnp.array(eq_tree_flat).all(axis=0) #& (self._p > 0) # ignores levels with no support
            level_idx = jax.lax.select(eq_mask.any(), eq_mask.argmax(), -1)
        else:
            level_idx = -1
        
        return obs, AutoResetFiniteState(env_state=env_state, level_idx=level_idx)
    
    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)