# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[mujoco]"
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4  \
    --command "uv run python cleanrl/ppo_continuous_action.py --no_cuda --track --capture_video --total_timesteps 5000000" \
    --num-seeds 1 \
    --workers 2 

OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids dm_control/cheetah-run-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0   \
    --command "uv run python cleanrl/ppo_continuous_action.py --no_cuda --track --capture_video " \
    --num-seeds 1 \
    --workers 2 
