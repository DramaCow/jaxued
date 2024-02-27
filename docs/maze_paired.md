This implements PAIRED.

See the DR example for details about what this outputs, and how to run it.

## Arguments

Argument Name  | Description | Default
-------------                   | ------------- | -------------
`--project`                     | Wandb project                                                                                                                              | todo
`--run_name`                    | This controls where the checkpoints are stored                                                                                                                                | None
`--seed`                        | Random seed                                                                                                                                | 0
`--mode`                        | "train" or "eval"                                                                                                                          | train
`--checkpoint_directory`        | Only valid if mode==eval where to load checkpoint from                                                                                     | None
`--checkpoint_to_eval`          | Only valid if mode==eval. This is the timestep to load from the above checkpoint directory                                                 | -1
`--checkpoint_save_interval`    | How often to save checkpoints                                                                                                              | 0
`--max_number_of_checkpoints`   | How many checkpoints to save in total                                                                                                      | 60
`--eval_freq`                   | How often to evaluate the agent and log                                                                                                    | 250
`--eval_num_attempts`           | How many attempts (episodes) per level to run for evaluation                                                                               | 10
`--eval_levels`                 | The eval levels to use                                                                                                                     | "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped", "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3"
`--num_updates`                 | Number of updates. Mutually exclusive with `num_env_steps`. Generally, `num_env_steps = num_updates * num_steps * num_train_envs`          | 30000
`--num_env_steps`               | Number of env steps. Mutually exclusive with `num_updates``                                                                                | None
`--num_train_envs`              | Number of training environments                                                                                                            | 32
`--student_lr`                          | Student's learning rate                                                                                                                  | 1e-4
`--student_max_grad_norm`               | Student's max PPO grad norm                                                                                                              | 0.5
`--student_num_steps`                   | Student's number of PPO rollout steps                                                                                                                | 256
`--student_num_minibatches`             | Student's number of PPO minibatches                                                                                                                  | 1
`--student_gamma`                       | Student's discount factor                                                                                                                            | 0.995
`--student_epoch_ppo`                   | Student's number of PPO epochs                                                                                                                       | 5
`--student_clip_eps`                    | Student's PPO Epsilon Clip                                                                                                                           | 0.2
`--student_gae_lambda`                  | Student's PPO Lambda                                                                                                                                 | 0.98
`--student_entropy_coeff`               | Student's PPO entropy coefficient                                                                                                                    | 1e-3
`--student_critic_coeff`                | Student's Critic coefficient  
`--adv_lr`                              | Adversary's learning rate                                                                                                                  | 1e-4
`--adv_max_grad_norm`                   | Adversary's max PPO grad norm                                                                                                              | 0.5
`--adv_num_steps`                       | Adversary's number of PPO rollout steps                                                                                                    | 50
`--adv_num_minibatches`                 | Adversary's number of PPO minibatches                                                                                                      | 1
`--adv_gamma`                           | Adversary's discount factor                                                                                                                | 0.995
`--adv_epoch_ppo`                       | Adversary's number of PPO epochs                                                                                                           | 5
`--adv_clip_eps`                        | Adversary's PPO Epsilon Clip                                                                                                               | 0.2
`--adv_gae_lambda`                      | Adversary's PPO Lambda                                                                                                                     | 0.98
`--adv_entropy_coeff`                   | Adversary's PPO entropy coefficient                                                                                                        | 1e-3
`--adv_critic_coeff`                    | Adversary's Critic coefficient                                                                                                             | 0.5
`--agent_view_size`             | The number of tiles the agent can see in front of it                                                                                       | 5
`--n_walls`                     | Number of walls to generate                                                                                                                | 25
