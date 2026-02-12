# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[mujoco]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4 Pusher-v4 \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track --capture_video" \
    --num-seeds 1 \
    --workers 2

uv pip install ".[mujoco, dm_control]"
uv run python -m cleanrl_utils.benchmark \
    --env-ids dm_control/cartpole-balance-v0 dm_control/cheetah-run-v0 dm_control/dog-walk-v0 dm_control/dog-run-v0 dm_control/hopper-hop-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/humanoid-run_pure_state-v0 dm_control/humanoid_CMU-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 \
    --command "uv run python cleanrl/vmpo_continuous_action.py --track" \
    --num-seeds 1 \
    --workers 2
