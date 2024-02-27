This example implements several variations of curation-based UED methods. In particular, it supports Domain Randomisation (DR), Prioritized Level Replay (PLR), Robust PLR, and ACCEL.

See the DR example for details about what this outputs, and how to run it.

## Arguments
Name  | Description | Default
-------------                   | ------------- | -------------
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
`--use_accel`                   | If True, runs ACCEL.                                                                                                                       | False
`--num_edits`                   | Only if `--use_accel=True`, the number of mutations done.                                                                                  | 5
`--project`                     | Wandb project                                                                                                                              | todo
`--run_name`                    | TODO ignore                                                                                                                                | todo
`--seed`                        | Random seed                                                                                                                                | 0
`--mode`                        | "train" or "eval"                                                                                                                          | train
`--checkpoint_directory`        | Only valid if mode==eval where to load checkpoint from                                                                                     | None
`--checkpoint_to_eval`          | Only valid if mode==eval. This is the timestep to load from the above checkpoint directory                                                 | -1
`--checkpoint_save_interval`    | How often to save checkpoints                                                                                                              | 0
`--max_number_of_checkpoints`   | How many checkpoints to save in total                                                                                                      | 60
`--eval_freq`                   | How often to evaluate the agent and log                                                                                                    | 250
`--eval_num_attempts`           | How many attempts (episodes) per level to run for evaluation                                                                               | 10
`--eval_levels`                 | The eval levels to use                                                                                                                     | "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped", "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3"
`--lr`                          | The agent's learning rate                                                                                                                  | 1e-4
`--max_grad_norm`               | The agent's max PPO grad norm                                                                                                              | 0.5
`--num_updates`                 | Number of updates. Mutually exclusive with `num_env_steps`. Generally, `num_env_steps = num_updates * num_steps * num_train_envs`          | 30000
`--num_env_steps`               | Number of env steps. Mutually exclusive with `num_updates``                                                                                | None
`--num_steps`                   | Number of PPO rollout steps                                                                                                                | 256
`--num_train_envs`              | Number of training environments                                                                                                            | 32
`--num_minibatches`             | Number of PPO minibatches                                                                                                                  | 1
`--gamma`                       | Discount factor                                                                                                                            | 0.995
`--epoch_ppo`                   | Number of PPO epochs                                                                                                                       | 5
`--clip_eps`                    | PPO Epsilon Clip                                                                                                                           | 0.2
`--gae_lambda`                  | PPO Lambda                                                                                                                                 | 0.98
`--entropy_coeff`               | PPO entropy coefficient                                                                                                                    | 1e-3
`--critic_coeff`                | Critic coefficient                                                                                                                         | 0.5
`--agent_view_size`             | The number of tiles the agent can see in front of it                                                                                       | 5
`--n_walls`                     | Number of walls to generate                                                                                                                | 25

