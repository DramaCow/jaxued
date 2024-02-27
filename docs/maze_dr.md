This example implements domain randomisation (DR). 
It uses the `AutoReset` wrapper to automatically reset the environment to a new random level upon episode termination. 
It follows the [PureJaxRL](https://github.com/luchris429/purejaxrl)-style training loop.


## Outputs
This code saves checkpoints to `./checkpoints/<run_name>/<seed>/models/<update_step>`, and if `mode=eval`, it saves results in `./results` in a `.npz` format.

## Usage

```bash
python examples/maze_dr.py <args>
```

## Examples
Run training:

```bash
python examples/maze_dr.py --run_name my_dr_test --project my_wandb_project --seed 0 --num_updates 10000
```

After that has finished, run evaluation on the final checkpoint.
```bash
python examples/maze_dr.py --mode eval --checkpoint_directory checkpoints/my_dr_test/0 --checkpoint_to_eval=-1
```

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