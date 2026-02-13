from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
import torch


@torch.no_grad()
def evaluate(
    agent: Any,
    make_env: Callable,
    env_id: str,
    seed: int,
    n_episodes: int = 10,
    run_name: str = "eval",
    capture_video: bool = False,
    gamma: float | None = None,
    vectorized: bool = False,
    return_episode_returns: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], list[float]]:
    if vectorized:
        if gamma is None:
            raise ValueError("gamma must be provided for vectorized MPO/VMPO evaluation.")

        returns: list[float] = []
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, i, capture_video, run_name, gamma) for i in range(n_episodes)]
        )
        try:
            obs, _ = eval_envs.reset(seed=seed)

            while len(returns) < n_episodes:
                action, _, _, _ = agent.act(obs)
                action = np.clip(
                    action,
                    eval_envs.single_action_space.low,
                    eval_envs.single_action_space.high,
                )
                next_obs, _, _, _, infos = eval_envs.step(action)
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            returns.append(
                                float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
                            )
                            if len(returns) >= n_episodes:
                                break
                obs = next_obs
        finally:
            eval_envs.close()
    else:
        if gamma is None:
            raise ValueError(
                "gamma must be provided for non-vectorized evaluation when make_env uses reward normalization."
            )
        returns = []
        eval_env = make_env(
            env_id,
            idx=0,
            capture_video=capture_video,
            run_name=run_name,
            gamma=gamma,
        )()
        try:
            for i in range(n_episodes):
                obs, _ = eval_env.reset(seed=seed + i)
                done = False
                while not done:
                    action = agent.act(obs)
                    obs, _, terminated, truncated, info = eval_env.step(action)
                    done = bool(terminated or truncated)
                if "episode" in info:
                    returns.append(
                        float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
                    )
                else:
                    raise RuntimeError(
                        "Missing episode statistics in evaluation info; ensure RecordEpisodeStatistics is enabled."
                    )
        finally:
            eval_env.close()

    returns_arr = np.asarray(returns, dtype=np.float32)
    eval_metrics = {
        "eval/return_mean": float(np.mean(returns_arr)),
        "eval/return_std": float(np.std(returns_arr)),
        "eval/return_min": float(np.min(returns_arr)),
        "eval/return_max": float(np.max(returns_arr)),
    }
    if return_episode_returns:
        return eval_metrics, returns
    return eval_metrics
