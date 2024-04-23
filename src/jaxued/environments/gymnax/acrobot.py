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
    dt: float                 = 0.2
    link_length_1: float      = 1.0
    link_length_2: float      = 1.0
    link_mass_1: float        = 1.0
    link_mass_2: float        = 1.0
    force_multiplier: float   = 1.0

@struct.dataclass
class EnvState:
    joint_angle1: float
    joint_angle2: float
    velocity_1:   float
    velocity_2:   float
    time:         int
    level_params: Level


@struct.dataclass
class EnvParams:
    available_torque: chex.Array
    link_com_pos_1: float = 0.5
    link_com_pos_2: float = 0.5
    link_moi: float = 1.0
    max_vel_1: float = 4 * jnp.pi
    max_vel_2: float = 9 * jnp.pi
    torque_noise_max: float = 0.0
    max_steps_in_episode: int = 500


class Acrobot(UnderspecifiedEnv):
    """
    JAX Compatible version of Acrobot-v1 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
    Note that we only implement the default 'book' version.
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (6,)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(available_torque=jnp.array([-1.0, 0.0, +1.0]))

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        torque = params.available_torque[action] * state.level_params.force_multiplier
        # Add noise to force action - always sample - conditionals in JAX
        torque = torque + jax.random.uniform(
            rng,
            shape=(),
            minval=-params.torque_noise_max,
            maxval=params.torque_noise_max,
        )

        # Augment state with force action so it can be passed to ds/dt
        s_augmented = jnp.array(
            [
                state.joint_angle1,
                state.joint_angle2,
                state.velocity_1,
                state.velocity_2,
                torque,
            ]
        )
        ns = rk4(s_augmented, params, state.level_params)
        joint_angle1 = wrap(ns[0], -jnp.pi, jnp.pi)
        joint_angle2 = wrap(ns[1], -jnp.pi, jnp.pi)
        velocity_1 = jnp.clip(ns[2], -params.max_vel_1, params.max_vel_1)
        velocity_2 = jnp.clip(ns[3], -params.max_vel_2, params.max_vel_2)

        done_angle = (
            -jnp.cos(joint_angle1) - jnp.cos(joint_angle2 + joint_angle1) > 1.0
        )
        reward = -1.0 * (1 - done_angle)

        # Update state dict and evaluate termination conditions
        state = EnvState(
            joint_angle1,
            joint_angle2,
            velocity_1,
            velocity_2,
            state.time + 1,
            level_params=state.level_params,
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {}
            # {"discount": self.discount(state, params)},
        )

    def reset_env_to_level(self, rng: chex.PRNGKey, level: Level, params: EnvParams) -> tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(
            rng, shape=(4,), minval=-0.1, maxval=0.1
        )
        state = EnvState(
            joint_angle1=init_state[0],
            joint_angle2=init_state[1],
            velocity_1=init_state[2],
            velocity_2=init_state[3],
            time=0,
            level_params=level
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.array(
            [
                jnp.cos(state.joint_angle1),
                jnp.sin(state.joint_angle1),
                jnp.cos(state.joint_angle2),
                jnp.sin(state.joint_angle2),
                state.velocity_1,
                state.velocity_2,
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination and construct updated state
        done_angle = (
            -jnp.cos(state.joint_angle1)
            - jnp.cos(state.joint_angle2 + state.joint_angle1)
            > 1.0
        )
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_angle, done_steps)
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Acrobot-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                params.max_vel_1,
                params.max_vel_2,
            ],
            dtype=jnp.float32,
        )
        return spaces.Box(-high, high, (6,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                jnp.finfo(jnp.float32).max,
                jnp.finfo(jnp.float32).max,
                params.max_vel_1,
                params.max_vel_2,
            ],
            dtype=jnp.float32,
        )
        return spaces.Dict(
            {
                "joint_angle1": spaces.Box(-high[0], high[0], (), jnp.float32),
                "joint_angle2": spaces.Box(-high[1], high[1], (), jnp.float32),
                "velocity_1": spaces.Box(-high[2], high[2], (), jnp.float32),
                "velocity_2": spaces.Box(-high[3], high[3], (), jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


def dsdt(s_augmented: chex.Array, t: float, params: EnvParams, level: Level) -> chex.Array:
    """Compute time derivative of the state change - Use for ODE int."""
    m1, m2 = level.link_mass_1, level.link_mass_2
    l1 = level.link_length_1
    lc1, lc2 = params.link_com_pos_1, params.link_com_pos_2
    I1, I2 = params.link_moi, params.link_moi
    g = 9.8
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1, theta2, dtheta1, dtheta2 = s
    d1 = (
        m1 * lc1 ** 2
        + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * jnp.cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * g * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2 ** 2 * jnp.sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * jnp.cos(theta1 - jnp.pi / 2)
        + phi2
    )
    ddtheta2 = (
        a
        + d2 / d1 * phi1
        - m2 * l1 * lc2 * dtheta1 ** 2 * jnp.sin(theta2)
        - phi2
    ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0])


def wrap(x: float, m: float, M: float) -> float:
    """For example, m = -180, M = 180 (degrees), x = 360 --> returns 0."""
    diff = M - m
    go_up = x < m     # Wrap if x is outside the left bound
    go_down = x >= M  # Wrap if x is outside OR on the right bound

    how_often = (
        go_up * jnp.ceil((m - x) / diff)           # if m - x is an integer, keep it
        + go_down * jnp.floor((x - M) / diff + 1)  # if x - M is an integer, round up
    )
    x_out = x - how_often * diff * go_down + how_often * diff * go_up
    return x_out


def rk4(y0: chex.Array, params: EnvParams, level: Level):
    """Runge-Kutta integration of ODE - Difference to OpenAI: Only 1 step!"""
    dt2 = level.dt / 2.0
    k1 = dsdt(y0, 0, params, level)
    k2 = dsdt(y0 + dt2 * k1, dt2, params, level)
    k3 = dsdt(y0 + dt2 * k2, dt2, params, level)
    k4 = dsdt(y0 + level.dt * k3, level.dt, params, level)
    yout = y0 + level.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout



def make_eval_levels_and_names():
    length     = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)
    masspole   = jnp.logspace(jnp.log10(0.05), jnp.log10(10), num=10)

    def get_arr(length, mass):
        return jnp.array([length, mass])
    
    def make_level(v):
        length, mass = v
        return Level(dt=0.2,
            link_length_1=length,
            link_length_2=length,
            link_mass_1=mass,
            link_mass_2=mass,
            force_multiplier=1.0,)
    arrs = jax.vmap(jax.vmap(get_arr, (0, None)), (None, 0))(length, masspole).reshape(-1, 2)

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
            dt=0.2,
            link_length_1=length,
            link_length_2=length,
            link_mass_1=mass,
            link_mass_2=mass,
            force_multiplier=1.0,
        ) # default
    return sample
    