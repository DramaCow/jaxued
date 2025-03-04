import numpy as np
import chex
from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Level:
    """This represents a level in the maze environment. The main features are the wall map, goal position, agent position and agent direction.
    """

    wall_map: chex.Array
    goal_pos: chex.Array
    agent_pos: chex.Array
    agent_dir: int
    width: int
    height: int
    
    def is_well_formatted(self):
        wall_map_is_binary = jnp.all((self.wall_map == 0) | (self.wall_map == 1))
        agent_goal_pos_distinct = ~(jnp.all(self.agent_pos == self.goal_pos))
        agent_dir_valid = (0 <= self.agent_dir) & (self.agent_dir <= 4)
        agent_not_on_wall = ~(self.wall_map[self.agent_pos[1], self.agent_pos[0]])
        goal_not_on_wall = ~(self.wall_map[self.goal_pos[1], self.goal_pos[0]])
        agent_within_bounds = (0 <= self.agent_pos[0]) & (self.agent_pos[0] < self.width) & (0 <= self.agent_pos[1]) & (self.agent_pos[1] < self.height)
        goal_within_bounds = (0 <= self.goal_pos[0]) & (self.goal_pos[0] < self.width) & (0 <= self.goal_pos[1]) & (self.goal_pos[1] < self.height)
        well_formatted = wall_map_is_binary & agent_goal_pos_distinct & agent_dir_valid & agent_not_on_wall & goal_not_on_wall & agent_within_bounds & goal_within_bounds
        return well_formatted
    
    @classmethod
    def from_str(cls, level_str):
        level_str = level_str.strip()
        rows = level_str.split('\n')
        nrows = len(rows)
        assert all(len(row) == len(rows[0]) for row in rows), "All rows must have same length"
        ncols = len(rows[0])
        
        wall_map = np.zeros((nrows, ncols), dtype=bool)
        goal_pos = []
        agent_pos = None
        agent_dir = None
        
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c == '#':
                    wall_map[y, x] = True
                elif c == 'G':
                    goal_pos.append((x, y))
                elif c == '>':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 0
                elif c == 'v':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 1
                elif c == '<':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 2
                elif c == '^':
                    assert agent_pos is None, "Agent position can only be set once."
                    agent_pos, agent_dir = (x, y), 3
                elif c == '.':
                    pass
                else:
                    raise Exception("Unexpected character.")
        
        assert len(goal_pos) > 0, "Goal position not set."
        assert agent_pos is not None, "Agent position not set."
        
        return Level(jnp.array(wall_map), *map(lambda x: jnp.array(x, dtype=jnp.uint32), (goal_pos[0], agent_pos)), jnp.array(agent_dir, dtype=jnp.uint8), ncols, nrows)
    
    def to_str(self):
        w, h = self.width, self.height
        h, w = self.wall_map.shape
        enc = np.full((h, w), None)
    
        for y in range(h):
            for x in range(w):
                enc[y, x] = '#' if self.wall_map[y, x] else '.'
        
        x, y = self.agent_pos
        if self.agent_dir == 0:
            agent_char = '>'
        elif self.agent_dir == 1:
            agent_char = 'v'
        elif self.agent_dir == 2:
            agent_char = '<'
        elif self.agent_dir == 3:
            agent_char = '^'    
        enc[y, x] = agent_char
        
        p = self.goal_pos
        x, y = self.goal_pos
        enc[y, x] = 'G'
        
        return '\n'.join([''.join(row) for row in enc]).strip()
    
    def pad_to_shape(self, max_width, max_height):
        batch_dims = self.wall_map.shape[:-2]
        h, w = self.wall_map.shape[-2:]
        assert max_width >= w and max_height >= h  
        new_wall_map = jax.lax.dynamic_update_slice(
            jnp.ones((*batch_dims, max_height, max_width), dtype=jnp.bool_),
            self.wall_map,
            (*(0,)*len(batch_dims), 0, 0),
        )
        return self.replace(wall_map=new_wall_map)
    
    @classmethod
    def stack(cls, levels):
        level_dims = np.array([[level.wall_map.shape[1], level.wall_map.shape[0]] for level in levels])
        max_width, max_height = level_dims.max(axis=0)
        return jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),
            *(level.pad_to_shape(max_width, max_height) for level in levels)
        )
    
    @classmethod
    def load_prefabs(cls, ids):
        return Level.stack([Level.from_str(prefabs[id]) for id in ids])

TrivialMaze = """
...
.#.
>#G
"""

TrivialMaze2 = """
.....
.....
..#..
..#..
>.#.G
"""

TrivialMaze3 = """
.......
.......
.......
...#...
...#...
...#...
>..#..G
"""
        
SixteenRooms = """
...#..#..#...
.>.......#...
...#..#......
#.###.##.###.
...#.........
......#..#...
##.#.##.###.#
...#.....#...
...#..#......
.####.##.#.##
...#..#..#...
......#....G.
...#.....#...
"""

SixteenRooms2 = """
...#.....#...
.>....#..#...
...#..#..#...
####.##.###.#
...#..#......
......#..#...
#.#####.#####
...#..#..#...
...#.........
##.##.##.####
...#..#..#...
......#....G.
...#..#..#...
"""

Labyrinth = """
.............
.###########.
.#.........#.
.#.#######.#.
.#.#.....#.#.
.#.#.###.#.#.
.#.#.#G#.#.#.
.#.#.#.#.#.#.
.#...#...#.#.
.#########.#.
.....#.....#.
####.#.#####.
>....#.......
"""

LabyrinthFlipped = """
.............
.###########.
.#.........#.
.#.#######.#.
.#.#.....#.#.
.#.#.###.#.#.
.#.#.#G#.#.#.
.#.#.#.#.#.#.
.#.#...#...#.
.#.#########.
.#.....#.....
.#####.#.####
.......#....<
"""

Labyrinth2 = """
>#...........
.#.#########.
.#.#.......#.
.#.#.#####.#.
.#.#.#...#.#.
...#.#.#.#.#.
####.#G#.#.#.
...#.###.#.#.
.#.#.....#.#.
.#.#######.#.
.#.........#.
.###########.
.............
"""

StandardMaze = """
.....#>...#..
.###.####.##.
.#...........
.########.###
........#....
######.#####.
....#..#.....
.##...##.####
..#.#..#...#.
#.#.##.###.#.
#.#..#...#...
#.##.###.###.
...#..G#.#...
"""

StandardMaze2 = """
...#.#....#..
.#.#.####...#
.#........#..
.########.###
...#..#.#.#.G
##.#.##.#.#..
>#.#....#.##.
.#.##.###..#.
.#..#..###.#.
.##.##.#.#.#.
.#...#.#.#.#.
.#.#.#.#.#.#.
...#...#.....
"""

StandardMaze3 = """
...>#.#......
.####.#.####.
.#....#.#....
...####.#.#.#
##.#....#.#..
...#.##.#.##.
.#.#.#..#..#G
.#.#.#.###.##
.#...#.#.#...
.###.#.#.###.
.#...#.#...#.
.#.###.#.#.#.
.#...#...#...
"""

prefabs = {
    "TrivialMaze": TrivialMaze.strip(),
    "TrivialMaze2": TrivialMaze2.strip(),
    "TrivialMaze3": TrivialMaze3.strip(),
    "SixteenRooms": SixteenRooms.strip(),
    "SixteenRooms2": SixteenRooms2.strip(),
    "Labyrinth": Labyrinth.strip(),
    "LabyrinthFlipped": LabyrinthFlipped.strip(),
    "Labyrinth2": Labyrinth2.strip(),
    "StandardMaze": StandardMaze.strip(),
    "StandardMaze2": StandardMaze2.strip(),
    "StandardMaze3": StandardMaze3.strip(),
}