python -m openrlbenchmark.rlops  --filters '?we=adrian-research&wpn=cleanrl-bench-3&ceik=env_id&cen=exp_name&metric=charts/episodic_return' 'vmpo_continuous_action' \
--env-ids HalfCheetah-v4 Walker2d-v4 Hopper-v4 Humanoid-v4  dm_control/cheetah-run-v0 dm_control/humanoid-walk-v0 dm_control/humanoid-run-v0 dm_control/walker-walk-v0 dm_control/walker-run-v0 --no-check-empty-runs  --pc.ncols 3  --pc.ncols-legend 2  --output-filename benchmark/cleanrl/vmpo_continuous_action  --scan-history

