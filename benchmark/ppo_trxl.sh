# export WANDB_ENTITY=openrlbenchmark

cd cleanrl/ppo_trxl
poetry install
OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids MortarMayhem-Grid-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --norm_adv --trxl_memory_length 119 --total_timesteps 100000000" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids MortarMayhem-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --reconstruction_coef 0.1 --trxl_memory_length 275" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids MysteryPath-Grid-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --trxl_memory_length 96 --total_timesteps 100000000" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids MysteryPath-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --trxl_memory_length 256" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids SearingSpotlights-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --reconstruction_coef 0.1 --trxl_memory_length 256" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids Endless-SearingSpotlights-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --reconstruction_coef 0.1 --trxl_memory_length 256 --total_timesteps 350000000" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids Endless-MortarMayhem-v0 Endless-MysteryPath-v0 \
    --command "python ./cleanrl/ppo_trxl/ppo_trxl.py --track --wandb_project_name cleanrl-bench-4  --trxl_memory_length 256 --total_timesteps 350000000" \
    --num-seeds 1 \
    --workers 5 
