python -m openrlbenchmark.rlops  --filters '?we=adrian-research&wpn=cleanrl-bench-3&ceik=env_id&cen=exp_name&metric=episode/r_mean' 'vmpo_trxl' \
--env-ids MortarMayhem-Grid-v0  MortarMayhem-v0  --pc.ncols 3  --pc.ncols-legend 2  --output-filename benchmark/cleanrl/vmpo_trxl  --scan-history

