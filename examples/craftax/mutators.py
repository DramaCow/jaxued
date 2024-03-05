import jax
import jax.numpy as jnp

from craftax_classic.envs.craftax_state import EnvState, StaticEnvParams
from craftax.constants import BlockType

def make_mutator_craftax_swap(static_env_params: StaticEnvParams, only_middle=True, is_craftax_classic=False):
    """Creates the unrestricted swap mutator. Here any tile can swap with any other tile.

    Args:
        static_env_params (StaticEnvParams): 
        only_middle (bool, optional): If true, swaps only happen in the middle 16 unit^2 area . Defaults to True.
        is_craftax_classic (bool, optional): If true, mutates a craftax classic env state, otherwise assumes the env state is Craftax. Defaults to False.
    """
    SIZE = static_env_params.map_size[0] * static_env_params.map_size[1]
    def add_blocktype(rng, level: EnvState, blocktype: BlockType) -> EnvState:
        vals = jnp.arange(SIZE)

        # Do not put something else on the player's position
        player_idx = level.player_position[0] * static_env_params.map_size[1] + level.player_position[1]
        probs = jnp.ones_like(vals).at[player_idx].set(0.0)
        
        if only_middle:
            temp = jnp.zeros_like(level.map[0], dtype=jnp.float32)
            mid_r, mid_c = [static_env_params.map_size[0] // 2, static_env_params.map_size[1] // 2]
            extent = 16 // 2
            min_r = mid_r - extent
            max_r = mid_r + extent

            min_c = mid_c - extent
            max_c = mid_c + extent
            
            temp = temp.at[min_r: max_r, min_c:max_c].set(1.0)
            probs = probs * temp.flatten()
        
        rng, _rng = jax.random.split(rng)
        position = jax.random.choice(_rng, vals, p=probs)

        row = position // static_env_params.map_size[1]
        col = position % static_env_params.map_size[1]

        # So we want to put blocktype at row, col. BUT, we want to keep the number of blocks the same. This means we need to take the current tile, and put it where the proposed tile is.

        tile_i_am_going_to_delete = level.map[0, row, col]
        block_check = (level.map[0] == blocktype).flatten()
        rng, _rng = jax.random.split(rng)
        def do_swap(level):
            # This chooses a random block that is the same as the one we want to add
            flat_idx = jax.random.choice(_rng, jnp.arange(SIZE), (), p=block_check)
            # And finds its position
            row_old = flat_idx // static_env_params.map_size[1]
            col_old = flat_idx % static_env_params.map_size[1]

            # Now, we swap these two.
            level = level.replace(map=level.map.at[0, row_old, col_old].set(tile_i_am_going_to_delete))
            level = level.replace(map=level.map.at[0, row, col].set(blocktype))
            return level

        def dont_swap(level):
            return level.replace(map=level.map.at[0, row, col].set(blocktype))
            
        does_new_block_exist_anywhere = block_check.any()
        
        level = jax.lax.cond(
            does_new_block_exist_anywhere, do_swap, dont_swap, level
        )

        return level.replace(map=level.map.at[0, row, col].set(blocktype))

    good_blocks = [
        BlockType.GRASS,
        BlockType.WATER,
        BlockType.STONE,
        BlockType.TREE,
        BlockType.COAL,
        BlockType.IRON,
        BlockType.DIAMOND,
        BlockType.LAVA,
        BlockType.RIPE_PLANT,
    ]
    NUM_BLOCKS_TO_CHOOSE = len(good_blocks)
    good_blocks_to_choose_from = jnp.array([b.value for b in good_blocks])
    def mutate_level(rng, level: EnvState, n=1):
        if is_craftax_classic: level = level.replace(map=level.map[None])
        def _single_mutate(carry, _):
            rng, level = carry
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            block_to_add = good_blocks_to_choose_from[jax.random.randint(_rng, (), 0, NUM_BLOCKS_TO_CHOOSE)]
            level = add_blocktype(_rng2, level, block_to_add)
            return (rng, level), None
        (rng, level), _ = jax.lax.scan(_single_mutate, (rng, level), None, length=n)
        if is_craftax_classic: level = level.replace(map=level.map[0])
        return level
    
    return mutate_level

def make_mutator_craftax_swap_restricted(static_env_params: StaticEnvParams, one_should_be_middle=False, is_craftax_classic=False):
    """Mutates levels by swapping two blocks of certain types.
        Stone can swap with ores and grass can swap with trees.

    Args:
        static_env_params (StaticEnvParams): 
        one_should_be_middle (bool, optional): If this is true, one of the swapped blocks should be in the middle 16 unit^2 part of the map Defaults to False.
        is_craftax_classic (bool, optional): If true, mutates a craftax classic env state, otherwise assumes the env state is Craftax. Defaults to False.
    """

    SIZE = static_env_params.map_size[0] * static_env_params.map_size[1]
    # 1. Pick a blocktype and a allowed swap one
    # 2. Pick a random block of that type
    def get_random_index_of_block(rng, level, blocktype_value, is_middle=False):
        block_check = (level.map[0] == blocktype_value).flatten()
        def answer1():
            if one_should_be_middle and is_middle:
                temp = jnp.zeros_like(level.map[0], dtype=jnp.float32)
                mid_r, mid_c = [static_env_params.map_size[0] // 2, static_env_params.map_size[1] // 2]
                extent = 16 // 2
                min_r = mid_r - extent
                max_r = mid_r + extent

                min_c = mid_c - extent
                max_c = mid_c + extent
                temp = temp.at[min_r: max_r, min_c:max_c].set(1.0)
                
                mult = block_check * temp.flatten()
                are_there_any_blocks_here = (mult).any()

                flat_idx = jax.lax.select(are_there_any_blocks_here, 
                                        jax.random.choice(rng, jnp.arange(SIZE), (), p=mult), 
                                        jax.random.choice(rng, jnp.arange(SIZE), (), p=block_check))
            else:
                flat_idx = jax.random.choice(rng, jnp.arange(SIZE), (), p=block_check)
            # And finds its position
            row = flat_idx // static_env_params.map_size[1]
            col = flat_idx % static_env_params.map_size[1]

            return row, col
        
        def answer2():
            return 0, 0
        
        # If there are no blocks of this type, we return 0, 0
        return jax.lax.cond(block_check.any(), answer1, answer2)
    
    def single_step(rng, level: EnvState):
        rng, _rng, _rng2, _rng3, _rng4, _rng5, _rng6, _rng7 = jax.random.split(rng, 8)
        block_to_add = good_blocks_to_choose_from[jax.random.randint(_rng, (), 0, NUM_BLOCKS_TO_CHOOSE)]

        # Tree and grass can swap
        new_block = BlockType.GRASS.value
        new_block = jax.lax.select(block_to_add == BlockType.GRASS.value, BlockType.TREE.value, new_block)
        new_block = jax.lax.select(block_to_add == BlockType.TREE.value, BlockType.GRASS.value, new_block)
        

        new_block = jax.lax.select(block_to_add == BlockType.STONE.value,   jax.random.choice(_rng2, for_stone), new_block)
        new_block = jax.lax.select(block_to_add == BlockType.COAL.value,    jax.random.choice(_rng3, for_coal), new_block)
        new_block = jax.lax.select(block_to_add == BlockType.DIAMOND.value, jax.random.choice(_rng4, for_diamond), new_block)
        new_block = jax.lax.select(block_to_add == BlockType.IRON.value,    jax.random.choice(_rng5, for_iron), new_block)

        # Now I need to find positions in the map where the two blocks are.

        r1, c1 = get_random_index_of_block(_rng6, level, block_to_add, is_middle=True)
        r2, c2 = get_random_index_of_block(_rng6, level, new_block,    is_middle=False)

        level = level.replace(map=level.map.at[0, r1, c1].set(new_block))
        level = level.replace(map=level.map.at[0, r2, c2].set(block_to_add))

        return (level)


    for_stone = jnp.array([b.value for b in   [BlockType.COAL, BlockType.IRON, BlockType.DIAMOND]])
    for_coal = jnp.array([b.value for b in    [BlockType.STONE, BlockType.IRON, BlockType.DIAMOND]])
    for_diamond = jnp.array([b.value for b in [BlockType.STONE, BlockType.IRON, BlockType.COAL]])
    for_iron = jnp.array([b.value for b in    [BlockType.STONE, BlockType.DIAMOND, BlockType.COAL]])
    good_blocks = [
        BlockType.GRASS,
        BlockType.STONE,
        BlockType.TREE,
        BlockType.COAL,
        BlockType.IRON,
        BlockType.DIAMOND,
    ]
    NUM_BLOCKS_TO_CHOOSE = len(good_blocks)
    good_blocks_to_choose_from = jnp.array([b.value for b in good_blocks])
    def mutate_level(rng, level: EnvState, n=1):
        if is_craftax_classic: level = level.replace(map=level.map[None])
        
        def _single_mutate(carry, _):
            rng, level = carry
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            level = single_step(_rng2, level)
            return (rng, level), None
        (rng, level), _ = jax.lax.scan(_single_mutate, (rng, level), None, length=n)
        if is_craftax_classic: level = level.replace(map=level.map[0])
        return level
    
    return mutate_level

def make_mutator_craftax_mutate_angles(generate_world, static_env_params: StaticEnvParams, params_to_use):
    """Mutates levels based on the angles of the fractal noise.

    Args:
        generate_world: Either Craftax classic's generate_world or Craftax's generate_world
        static_env_params (StaticEnvParams):
        params_to_use (_type_):
    """
    def mutate_level(rng, level: EnvState, n=1):
        rng, *rngs = jax.random.split(rng, 5)
        new_angles = jax.tree_map(lambda x, rr: jnp.clip(x + jax.random.uniform(rr, x.shape, minval=-0.2, maxval=0.2), 0, 1), level.fractal_noise_angles, tuple(rngs))
        return generate_world(rng, params_to_use.replace(fractal_noise_angles=new_angles), static_env_params).replace(fractal_noise_angles=new_angles)
    
    return mutate_level

