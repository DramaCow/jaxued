This example file implements DR, PLR and ACCEL for [Craftax](https://github.com/MichaelTMatthews/Craftax).


## Usage

```bash
python examples/craftax/maze_plr.py <args>
```

## Different Methods
### DR
`--exploratory_grad_updates --replay_prob 0.0`

### PLR
`--exploratory_grad_updates --replay_prob 0.5`

### Robust PLR
`--no-exploratory_grad_updates --replay_prob 0.5`

### ACCEL
One could have `--exploratory_grad_updates` on or off here.
`--replay_prob 0.5 --use_accel --num_edits 30 --accel_mutation <mut>`
where `<mut>` is either `swap`, `swap_restricted` or `noise`.

- `swap`: Swaps two random tiles in the map
- `swap_restricted`: Swaps two random tiles, but enforces which tiles can be swapped with each other; for instance, stone and ores can swap.
- `noise`: The mutation adds random numbers to the level's perlin noise vectors.


### Inner and Outer Rollouts
Due to the long episodes in Craftax, we have separate inner and outer rollout lengths. The inner rollout length is the time between PPO updates (e.g. `--num_steps` in PureJaxRL). The outer rollout length is how many of these inner rollouts should be used to calculate the score of a level. For instance, `--num_steps 64 --outer_rollout_steps 64` updates the policy every 64 steps, but the maximum episode length is 64*64 = 4096.

## Arguments

Name  | Description | Default
-------------                   | ------------- | -------------
`--project`                     | Wandb project                                                                                                                              | JAXUED_TEST
`--run_name`                    | This controls where the checkpoints are stored                                                                                                                                | None
`--seed`                        | Random seed                                                                                                                                | 0
`--mode`                        | "train" or "eval"                                                                                                                          | train
`--checkpoint_directory`        | Only valid if mode==eval where to load checkpoint from                                                                                     | None
`--checkpoint_to_eval`          | Only valid if mode==eval. This is the timestep to load from the above checkpoint directory                                                 | -1
`--checkpoint_save_interval`    | How often to save checkpoints                                                                                                              | 0
`--max_number_of_checkpoints`   | How many checkpoints to save in total                                                                                                      | 60
`--eval_freq`                   | How often to evaluate the agent and log                                                                                                    | 10
`--eval_num_attempts`           | How many attempts (episodes) per level to run for evaluation                                                                               | 1
`--n_eval_levels`               | How many (random) evaluation levels to use                                                                                                 | 5
`--num_eval_steps`              | Episode length during evaluation                                                                                                           | 2048
`--lr`                          | The agent's learning rate                                                                                                                  | 1e-4
`--max_grad_norm`               | The agent's max PPO grad norm                                                                                                              | 1.0
`--num_updates`                 | Number of updates. Mutually exclusive with `num_env_steps`. Generally, `num_env_steps = num_updates * num_steps * num_train_envs`          | 256
`--num_env_steps`               | Number of env steps. Mutually exclusive with `num_updates``                                                                                | None
`--num_steps`                   | Number of PPO rollout steps                                                                                                                | 64
`--outer_rollout_steps`         | Number of inner rollouts in one PLR/ACCEL step                                                                                             | 64
`--num_train_envs`              | Number of training environments                                                                                                            | 1024
`--num_minibatches`             | Number of PPO minibatches                                                                                                                  | 2
`--gamma`                       | Discount factor                                                                                                                            | 0.995
`--epoch_ppo`                   | Number of PPO epochs                                                                                                                       | 5
`--clip_eps`                    | PPO Epsilon Clip                                                                                                                           | 0.2
`--gae_lambda`                  | PPO Lambda                                                                                                                                 | 0.9
`--entropy_coeff`               | PPO entropy coefficient                                                                                                                    | 0.01
`--critic_coeff`                | Critic coefficient                                                                                                                         | 0.5
`--use_accel`                   | Should we use ACCEL                                                                                                                         | False
`--num_edits`                   | How many ACCEL edits should we make                                                                                                                         | 30
`--accel_mutation`              | What mutation to use, `swap`, `swap_restricted` or `noise`                                                                                                                         | "swap"
`--score_function`              | The score function to use, `pvl` or `MaxMC`                                                                                                | MaxMC
`--exploratory_grad_updates`    | If `True`, trains on random levels                                                                                                         | False
`--level_buffer_capacity`       | The maximum number of levels in the buffer.                                                                                                | 4000
`--replay_prob`                 | The probability of performing a `replay` step                                                                                              | 0.8
`--staleness_coeff`             | The coefficient used to combine staleness and score weights                                                                                | 0.3
`--temperature`                 | The temperature when using rank prioritization, only valid if `prioritization=rank`.                                                       | 0.3
`--topk_k`                      | The number of levels sampled when using `topk` prioritization. Only valid if `prioritization=topk`.                                        | 4
`--minimum_fill_ratio`          | The minimum number of environments in the level before replay can be triggered.                                                            | 0.5
`--prioritization`              | `rank` or `topk`.                                                                                                                          | rank
`--buffer_duplicate_check`      | If True, duplicate levels cannot be added to the buffer.                                                                                   | True