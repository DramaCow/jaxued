import jax
import jax.numpy as jnp
import chex
from .level import Level
from .env import DIR_TO_VEC
from enum import IntEnum
import numpy as np
from typing import Callable

def make_level_generator(height: int, width: int, n_walls: int) -> Callable[[chex.PRNGKey], Level]:
    """This takes in a height, width and number of walls and returns a function that takes in a PRNGKey and returns a level.

    Args:
        height (int): 
        width (int): 
        n_walls (int): 
    """
    def sample(rng: chex.PRNGKey) -> Level:
        max_w, max_h = width, height
        all_pos = jnp.arange(max_w * max_h, dtype=jnp.uint32)
        valid_mask = (all_pos % max_w < width) & (all_pos < max_w * height)
        
        rng_wall, rng_agent_pos, rng_agent_dir, rng_goal = jax.random.split(rng, 4)
        # n_walls = jax.random.choice(rng_n_walls, n_walls+1)
        
        choices = jax.random.choice(rng_wall, max_w*max_h, shape=(max_w*max_h,), p=valid_mask, replace=True)
        choices = jnp.where(all_pos < n_walls, choices, choices[0])
        occupied_mask = jnp.zeros(max_w * max_h, dtype=jnp.bool_).at[choices].set(n_walls > 0) | ~valid_mask
        wall_map = occupied_mask.reshape(max_h, max_w)

        # Reset agent position + dir
        agent_idx = jax.random.choice(rng_agent_pos, all_pos, shape=(1,), p=(~occupied_mask).astype(jnp.float32))
        occupied_mask = occupied_mask.at[agent_idx].set(True)
        agent_pos = jnp.array([agent_idx%max_w, agent_idx//max_w], dtype=jnp.uint32).flatten()

        # Reset agent direction
        agent_dir = jax.random.choice(rng_agent_dir, jnp.arange(len(DIR_TO_VEC), dtype=jnp.uint8))

        # Reset goal position
        goal_idx = jax.random.choice(rng_goal, all_pos, shape=(1,), p=(~occupied_mask).astype(jnp.float32))
        goal_pos = jnp.array([goal_idx%max_w, goal_idx//max_w], dtype=jnp.uint32).flatten()
        
        return Level(wall_map, goal_pos, agent_pos, agent_dir, width, height)
    
    return sample

def make_level_mutator(max_num_edits: int) -> Callable[[chex.PRNGKey, Level, int], Level]:
    def mutate(rng: chex.PRNGKey, level: Level, num_edits: int = 1) -> Level:
        max_w, max_h = level.wall_map.shape[1], level.wall_map.shape[0]
        all_pos = jnp.arange(max_w * max_h, dtype=jnp.uint32)
        valid_mask = (all_pos % max_w < level.width) & (all_pos < max_w * level.height)
        
        rng, rng_perm, rng_loc, rng_action, rng_goal, rng_agent, rng_agent_dir = jax.random.split(rng, 7)
        edit_locs = jax.random.permutation(rng_perm, jnp.unique(jax.random.choice(rng_loc, all_pos, shape=(max_num_edits,), p=valid_mask), size=max_num_edits, fill_value=-1))
        edit_locs = jnp.where(jnp.cumsum(edit_locs != -1) <= num_edits, edit_locs, -1) # mask out extra action
        actions = jax.random.choice(rng_action, 4, shape=(max_num_edits,))
        
        def mutation_step(carry, input):
            rng, level, agent_displaced, goal_displaced = carry
            edit_loc, action = input
            
            def on_mutate(rng, level, edit_loc, action, agent_displaced, goal_displaced):
                x, y = edit_loc%max_w, edit_loc//max_w
                
                agent_displaced = ((level.agent_pos[0] == x) & (level.agent_pos[1] == y)) | agent_displaced
                goal_displaced = ((level.goal_pos[0] == x) & (level.goal_pos[1] == y)) | goal_displaced

                def add_wall(rng):
                    return rng, level.replace(wall_map=level.wall_map.at[y, x].set(True)), agent_displaced, goal_displaced
                
                def remove_wall(rng):
                    return rng, level.replace(wall_map=level.wall_map.at[y, x].set(False)), agent_displaced, goal_displaced
                
                def move_agent_pos(rng):
                    rng, rng_dir = jax.random.split(rng)
                    agent_pos = jnp.array([x, y], dtype=jnp.uint32)
                    agent_dir = jnp.array(jax.random.choice(rng_dir, 4), dtype=jnp.uint8)
                    return rng, level.replace(wall_map=level.wall_map.at[y, x].set(False), agent_pos=agent_pos, agent_dir=agent_dir), False, goal_displaced
                
                def move_goal_pos(rng):
                    goal_pos = jnp.array([x, y], dtype=jnp.uint32)
                    return rng, level.replace(wall_map=level.wall_map.at[y, x].set(False), goal_pos=goal_pos), agent_displaced, False
                
                return jax.lax.switch(action, [
                    add_wall,
                    remove_wall,
                    move_agent_pos,
                    move_goal_pos
                ], rng)
                
            def do_nothing(rng, level, edit_loc, action, agent_displaced, goal_displaced):
                return rng, level, agent_displaced, goal_displaced
                
            return jax.lax.cond(edit_loc != -1, on_mutate, do_nothing, rng, level, edit_loc, action, agent_displaced, goal_displaced), None
        (rng, level, agent_displaced, goal_displaced), _ = jax.lax.scan(mutation_step, (rng, level, False, False), (edit_locs, actions))
        
        agent_idx = level.agent_pos[1] * max_w + level.agent_pos[0]
        goal_idx = level.goal_pos[1] * max_w + level.goal_pos[0]
        
        # handle displaced goal
        # NOTE: mark agent position as valid only if there is not a wall there AND agent has been displaced
        p = (~level.wall_map.flatten() & valid_mask).at[agent_idx].set(agent_displaced & ~(level.wall_map[level.agent_pos[1], level.agent_pos[0]]))
        new_goal_idx = jax.random.choice(rng_goal, all_pos, p=p)
        goal_idx = jax.lax.select(goal_displaced, new_goal_idx, goal_idx)
        goal_pos = jnp.array([goal_idx%max_w, goal_idx//max_w], dtype=jnp.uint32)
        
        # handle displaced agent
        p = (~level.wall_map.flatten() & valid_mask).at[goal_idx].set(False)
        new_agent_idx = jax.random.choice(rng_agent, all_pos, p=p)
        new_agent_dir = jnp.array(jax.random.choice(rng_agent_dir, 4), dtype=jnp.uint8)
        agent_idx = jax.lax.select(agent_displaced, new_agent_idx, agent_idx)
        agent_pos = jnp.array([agent_idx%max_w, agent_idx//max_w], dtype=jnp.uint32)
        agent_dir = jax.lax.select(agent_displaced, new_agent_dir, level.agent_dir)
        
        new_level = level.replace(goal_pos=goal_pos, agent_pos=agent_pos, agent_dir=agent_dir)
        
        # jax.lax.cond(~(new_level.is_well_formatted()), jax.debug.breakpoint, lambda: None)
        
        return new_level
    
    return mutate

# This function is a modified version of
# https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/envs/maze/maze_mutators.py.
# Credit: minimax
def make_level_mutator_minimax(max_num_edits: int) -> Callable[[chex.PRNGKey, Level, int], Level]:
    class Mutations(IntEnum):
        # Turn left, turn right, move forward
        NO_OP = 0
        FLIP_WALL = 1
        MOVE_GOAL = 2
    
    def flip_wall(rng, state):
        wall_map = state.wall_map
        h,w = wall_map.shape
        wall_mask = jnp.ones((h*w,), dtype=jnp.bool_)

        goal_idx = w*state.goal_pos[1] + state.goal_pos[0]
        agent_idx = w*state.agent_pos[1] + state.agent_pos[0]
        wall_mask = wall_mask.at[goal_idx].set(False)
        wall_mask = wall_mask.at[agent_idx].set(False)

        flip_idx = jax.random.choice(rng, np.arange(h*w), p=wall_mask)
        flip_y = flip_idx//w
        flip_x = flip_idx%w

        flip_val = ~wall_map.at[flip_y,flip_x].get()
        next_wall_map = wall_map.at[flip_y,flip_x].set(flip_val)

        return state.replace(wall_map=next_wall_map)


    def move_goal(rng, state):
        wall_map = state.wall_map
        h,w = wall_map.shape
        wall_mask = wall_map.flatten()

        goal_idx = w*state.goal_pos[1] + state.goal_pos[0]
        agent_idx = w*state.agent_pos[1] + state.agent_pos[0]
        wall_mask = wall_mask.at[goal_idx].set(True)
        wall_mask = wall_mask.at[agent_idx].set(True)

        next_goal_idx = jax.random.choice(rng, np.arange(h*w), p=~wall_mask)
        next_goal_y = next_goal_idx//w
        next_goal_x = next_goal_idx%w

        next_wall_map = wall_map.at[next_goal_y,next_goal_x].set(False)
        next_goal_pos = jnp.array([next_goal_x,next_goal_y], dtype=jnp.uint32)

        return state.replace(wall_map=next_wall_map, goal_pos=next_goal_pos)

    def move_goal_flip_walls(rng, level, n=1):
        def _mutate(carry, step):
            state = carry
            rng, mutation = step

            def _apply(rng, state):    
                rng, arng, brng = jax.random.split(rng, 3)

                is_flip_wall = jnp.equal(mutation, Mutations.FLIP_WALL.value)
                mutated_state = flip_wall(arng, state)
                next_state = jax.tree_util.tree_map(lambda x,y: jax.lax.select(is_flip_wall, x, y), mutated_state, state)

                is_move_goal = jnp.equal(mutation, Mutations.MOVE_GOAL.value)
                mutated_state = move_goal(brng, state)
                next_state = jax.tree_util.tree_map(lambda x,y: jax.lax.select(is_move_goal, x, y), mutated_state, next_state)
                
                return next_state
                
            return jax.lax.cond(mutation != -1, _apply, lambda *_: state, rng, state), None

        rng, nrng, *mrngs = jax.random.split(rng, max_num_edits+2)
        mutations = jax.random.choice(nrng, np.arange(len(Mutations)), (max_num_edits,))
        mutations = jnp.where(jnp.arange(max_num_edits) < n, mutations, -1) # mask out extra mutations

        new_level, _ = jax.lax.scan(_mutate, level, (jnp.array(mrngs), mutations))

        return new_level
    
    return move_goal_flip_walls