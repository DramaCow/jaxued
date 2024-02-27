This is a maze environment based on [Minigrid](https://github.com/Farama-Foundation/Minigrid). The overall task is partially observable navigation, and the agent must move around in a maze to reach the goal.

## Action Space
7 dimensional discrete action space, where only the first three are used. We do not mask out invalid actions.

## Observation Space
The size of the observation depends on the agent's view size (`k`), which is by default 5. The observations is `kxkx3`, consisting of floats, indicating which tile is in each of the cells around the agent.

## Reward Function
By default, `penalize_time` is `True`, which means the reward upon reaching the goal is `1 - 0.9 * (t / T)`, where `t` is teh current timestep, and `T` is the maximum number of timesteps in the episode.
If `penalize_time` is `False`, the reward is 1 upon reaching the goal and zero otherwise.

## Level Generation
The function `make_level_generator` in `jaxued.environments.maze.util` returns a callable that generates a maze given an `rng`.

## Rendering
We also provide JITted rendering functionality in `jaxued.environments.maze.renderer`.
This can be used as follows:

```python
env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
sample_random_level = make_level_generator(env.max_height, env.max_width, 25)
env_renderer = MazeRenderer(env, tile_size=8)
obs, state = env.reset_to_level(rng, sample_random_level(rng), env_params)

image = env_renderer.render_state(state)
plt.imshow(image)
```