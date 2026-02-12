# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[mujoco]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4  \
    --command "uv run python cleanrl/mpo_continuous_action.py --track --capture_video  --wandb_project_name cleanrl-bench-3" \
    --num-seeds 2 \
    --workers 2

uv pip install ".[mujoco, dm_control]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/cheetah-run-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --command "uv run python cleanrl/mpo_continuous_action.py  --track  --wandb_project_name cleanrl-bench-3" \
    --num-seeds 2 \
    --workers 2

