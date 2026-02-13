# export WANDB_ENTITY=openrlbenchmark

# cd cleanrl/vmpo_trxl
# poetry install
OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids MortarMayhem-Grid-v0 \
    --command "python ./cleanrl/vmpo_trxl/vmpo_trxl.py --track --norm_adv --trxl_memory_length 119 --total_timesteps 300000000 --wandb_project_name cleanrl-bench-4" \
    --num-seeds 1 \
    --workers 5 

OMP_NUM_THREADS=4 uv run python -m cleanrl_utils.benchmark \
    --env-ids MysteryPath-Grid-v0 \
    --command "python ./cleanrl/vmpo_trxl/vmpo_trxl.py --track --trxl_memory_length 96 --total_timesteps 300000000 --wandb_project_name cleanrl-bench-4" \
    --num-seeds 1 \
    --workers 5 
