# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[mujoco]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4  \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --capture_video --wandb_project_name cleanrl-vmpo-4 --total_timesteps 5000000 --num_envs 8" \
    --num-seeds 1 \
    --workers 2 

uv pip install ".[mujoco, dm_control]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/cheetah-run-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --wandb_project_name cleanrl-vmpo-4 --total_timesteps 5000000 --num_envs 8" \
    --num-seeds 1 \
    --workers 2 

# DAE
uv pip install ".[mujoco]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4  \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --capture_video --wandb_project_name cleanrl-vmpo-4-dae --total_timesteps 5000000 --advantage_estimator dae  --num_envs 8" \
    --num-seeds 1 \
    --workers 2 

uv pip install ".[mujoco, dm_control]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/cheetah-run-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --wandb_project_name cleanrl-vmpo-4-dae --total_timesteps 5000000 --advantage_estimator dae   --num_envs 8" \
    --num-seeds 1 \
    --workers 2 

#SGD 
uv run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4  \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --capture_video --wandb_project_name cleanrl-vmpo-4-gae-sgd --total_timesteps 5000000 --advantage_estimator gae  --optimizer sgd  --num_envs 8" \
    --num-seeds 1 \
    --workers 2

uv pip install ".[mujoco, dm_control]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/cheetah-run-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --wandb_project_name cleanrl-vmpo-4-gae-sgd --total_timesteps 5000000 --advantage_estimator gae  --optimizer sgd  --num_envs 8" \
    --num-seeds 1 \
    --workers 2 