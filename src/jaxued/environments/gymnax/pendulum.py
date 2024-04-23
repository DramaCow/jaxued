import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Callable, Tuple, Optional
import chex
from flax import struct

from jaxued.environments.underspecified_env import UnderspecifiedEnv

@struct.dataclass
class Level:
    dt: float = 0.05
    g: float = 10.0  # gravity
    m: float = 1.0  # mass
    l: float = 1.0  # length

@struct.dataclass
class EnvState:
    theta: float
    theta_dot: float
    last_u: float  # Only needed for rendering
    time: int
    level_params: Level


@struct.dataclass
class EnvParams:
    max_speed: float = 8.0
    max_torque: float = 2.0
    max_steps_in_episode: int = 200


class Pendulum(UnderspecifiedEnv):
    """
    JAX Compatible version of Pendulum-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (3,)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Integrate pendulum ODE and return transition."""
        u = jnp.clip(action, -params.max_torque, params.max_torque)
        reward = -(
            angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot ** 2
            + 0.001 * (u ** 2)
        )
        reward = reward.squeeze()

        newthdot = state.theta_dot + (
            (
                3 * state.level_params.g / (2 * state.level_params.l) * jnp.sin(state.theta)
                + 3.0 / (state.level_params.m * state.level_params.l ** 2) * u
            )
            * state.level_params.dt
        )

        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * state.level_params.dt

        # Update state dict and evaluate termination conditions
        state = EnvState(
            newth.squeeze(), newthdot.squeeze(), u.reshape(), state.time + 1,
            level_params=state.level_params
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {},
        )

    def reset_env_to_level(self, rng: chex.PRNGKey, level: Level, params: EnvParams) -> tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(rng, shape=(2,), minval=-high, maxval=high)
        state = EnvState(theta=state[0], theta_dot=state[1], last_u=0.0, time=0, level_params=level)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                state.theta_dot,
            ]
        ).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([1.0, 1.0, params.max_speed], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(3,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "theta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def make_eval_levels_and_names():
    length     = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)
    mass   = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)

    def get_arr(length, mass):
        return jnp.array([length, mass])
    
    def make_level(v):
        length, mass = v
        return Level(l=length, m=mass)
    
    arrs = jax.vmap(jax.vmap(get_arr, (0, None)), (None, 0))(length, mass).reshape(-1, 2)

    levels = jax.vmap(make_level)(arrs)
    default = Level()
    levels = jax.tree_map(lambda x, new: jnp.concatenate([x, jnp.array(new)[None]], axis=0), levels, default)
    return levels, [f"length_{i:<2}_mass_{j:<2}" for i, j in arrs] + ['default']
    


def make_level_generator() -> Callable[[chex.PRNGKey], Level]:
    def sample(rng: chex.PRNGKey) -> Level:
        rng1, rng2 = jax.random.split(rng)
        length = jax.random.uniform(rng1) * 10 + 0.01
        mass = jax.random.uniform(rng2) * 10 + 0.01
        return Level(
            l=length,
            m=mass,
            g=10.0,
            dt=0.05,
        ) # default
    return sample
