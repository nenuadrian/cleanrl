from typing import Any, Callable, Dict

import gymnasium as gym
import numpy as np
import torch


@torch.no_grad()
def evaluate(
    agent: Any,
    make_env: Callable,
    flatten_obs: Callable[[Any], np.ndarray],
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
            episode_returns = np.zeros(n_episodes, dtype=np.float32)
            dones = np.zeros(n_episodes, dtype=bool)

            while len(returns) < n_episodes:
                obs = flatten_obs(obs)
                action, _, _, _ = agent.act(obs, deterministic=True)
                action = np.clip(
                    action,
                    eval_envs.single_action_space.low,
                    eval_envs.single_action_space.high,
                )
                next_obs, reward, terminated, truncated, _ = eval_envs.step(action)
                episode_returns += reward

                done = np.asarray(terminated) | np.asarray(truncated)
                for i in range(n_episodes):
                    if not dones[i] and done[i]:
                        returns.append(float(episode_returns[i]))
                        dones[i] = True
                obs = next_obs
        finally:
            eval_envs.close()
    else:
        returns = []
        eval_env = make_env(
            env_id,
            seed=seed,
            idx=0,
            capture_video=capture_video,
            run_name=run_name,
        )()
        try:
            for i in range(n_episodes):
                obs, _ = eval_env.reset(seed=seed + i)
                obs = flatten_obs(obs)
                done = False
                ep_return = 0.0
                while not done:
                    action = agent.act(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    obs = flatten_obs(obs)
                    ep_return += float(reward)
                    done = bool(terminated or truncated)
                returns.append(ep_return)
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
