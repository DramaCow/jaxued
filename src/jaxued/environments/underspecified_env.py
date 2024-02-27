from typing import Any, Tuple, Optional, Union
import jax
import chex
from functools import partial
from flax import struct

@struct.dataclass
class EnvState:
    pass

@struct.dataclass
class Observation:
    pass

@struct.dataclass
class Level:
    pass

@struct.dataclass
class EnvParams:
    pass

class UnderspecifiedEnv(object):
    """
    The UnderspecifiedEnv class defines a UPOMDP, and acts similarly to (but not identically to) a Gymnax environment.

    The UnderspecifiedEnv class has the following interface:
        * `params = env.default_params`
        * `action_space = env.action_space(params)`
        * `obs, state = env.reset_to_level(rng, level, params)`
        * `obs, state, reward, done, info = env.step(rng, state, action, params)`

    Every environment must implement only the following methods:
        * `step_env`: Perform a step of the environment
        * `reset_env_to_level`: Reset the environment to a particular level
        * `action_space`: Return the action space of the environment
            
    The environment also does not automatically reset to a new level once the environment has restarted. 
    Look at the `AutoReplay` wrapper if this is desired.
    """
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[Observation, EnvState, float, bool, dict]:
        if params is None:
            params = self.default_params
        return self.step_env(rng, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: Optional[EnvParams] = None
    ) -> Tuple[Observation, EnvState]:
        if params is None:
            params = self.default_params
        return self.reset_env_to_level(rng, level, params)

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        raise NotImplementedError

    def reset_env_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[Observation, EnvState]:
        raise NotImplementedError

    def action_space(self, params: EnvParams) -> Any:
        raise NotImplementedError