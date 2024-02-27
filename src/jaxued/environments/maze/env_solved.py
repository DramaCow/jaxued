from .env import EnvParams, Maze
import chex
from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class EnvState:
    agent_pos: chex.Array
    agent_dir: int
    goal_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool
    min_steps: chex.Array

class MazeSolved(Maze):
    """
    An extension of the Maze environment where the shortest path to goal
    for each state has been precomputed. This is therefore useful for computing
    the true optimal value, which it turn can be used for computing the true
    regret.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def init_state_from_level(self, level):
        state = super().init_state_from_level(level)
        return EnvState(
            agent_pos=state.agent_pos,
            agent_dir=state.agent_dir,
            goal_pos=state.goal_pos,
            wall_map=state.wall_map,
            maze_map=state.maze_map,
            time=state.time,
            terminal=state.terminal,
            min_steps=self._precompute_min_steps_to_goal(level),
        )
        
    def min_steps_to_goal(self, state: EnvState):
        return state.min_steps[state.agent_dir, state.agent_pos[1], state.agent_pos[0]]
    
    def optimal_value(self, state: EnvState, gamma: float, params: EnvParams):
        n = self.min_steps_to_goal(state)
        if self.penalize_time:
            N = state.time + n
            value = (1.0 - 0.9*((N+1)/params.max_steps_in_episode)) * gamma ** n
        else:
            value = 1.0 * gamma ** n
        return jnp.where(n != jnp.inf, value, 0)
    
    def is_solveable(self, state: EnvState, params: EnvParams):
        return self.min_steps_to_goal(state) != jnp.inf
    
    def _precompute_min_steps_to_goal(self, level):
        wall_values = jnp.repeat(jnp.where(level.wall_map, jnp.inf, -jnp.inf)[None, ...], 4, axis=0)
        
        def compute_next(values):
            fwd_values = jnp.array([
                jnp.roll(values[0], -1, axis=1).astype(float).at[:,-1].set(jnp.inf),
                jnp.roll(values[1], -1, axis=0).astype(float).at[-1,:].set(jnp.inf),
                jnp.roll(values[2], 1, axis=1).astype(float).at[:,0].set(jnp.inf),
                jnp.roll(values[3], 1, axis=0).astype(float).at[0,:].set(jnp.inf),
            ])
            new_values = jnp.empty_like(values)
            for i in range(4):
                new_values = new_values.at[i].set(jnp.min(
                    jnp.array([values[i], values[i-1] + 1, values[(i+1)%4] + 1, fwd_values[i] + 1]), axis=0
                ))
            return jnp.maximum(new_values, wall_values)
        
        def cond_fn(carry):
            values, next_values = carry
            return jnp.any(values != next_values)
        
        def body_fn(carry):
            _, values = carry
            return values, compute_next(values)
        
        values = jnp.full((4, self.max_height, self.max_width), jnp.inf).at[:, level.goal_pos[1], level.goal_pos[0]].set(0)
        return jax.lax.while_loop(cond_fn, body_fn, (values, compute_next(values)))[0]