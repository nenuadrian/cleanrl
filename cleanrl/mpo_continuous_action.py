# docs and experiment results can be found at:
# https://docs.cleanrl.dev/
#
# This script is a CleanRL-style runner for the project's MPO implementation.
# It keeps the MPO update logic from trainers/mpo while exposing a single-file
# experiment entrypoint with tyro args, tensorboard logging, and optional wandb.
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

# Ensure repo-root modules are importable when running from inside /cleanrl.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainers.mpo.agent import MPOAgent, MPOConfig  # noqa: E402
from trainers.mpo.replay_buffer import MPOReplayBuffer  # noqa: E402
from utils.env import flatten_obs, infer_obs_dim, make_env as make_base_env  # noqa: E402


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder at the end"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    update_after: int = 10_000
    """number of env steps before starting updates"""
    batch_size: int = 512
    """batch size sampled from replay buffer"""
    updates_per_step: int = 1
    """number of gradient updates per environment step"""
    eval_interval: int = 7_000
    """evaluate every N steps; 0 disables eval"""
    eval_episodes: int = 10
    """number of episodes used for each evaluation"""
    save_interval: int = 50_000
    """save checkpoint every N steps; 0 disables periodic save"""

    policy_layer_sizes: tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for policy network"""
    critic_layer_sizes: tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for critic networks"""

    gamma: float = 0.995
    """discount factor"""
    tau: float = 0.005
    """soft target update factor"""
    policy_lr: float = 3e-4
    """policy optimizer learning rate"""
    q_lr: float = 3e-4
    """critic optimizer learning rate"""
    temperature_init: float = 1.0
    """initial MPO temperature (eta)"""
    temperature_lr: float = 3e-4
    """temperature optimizer learning rate"""
    kl_epsilon: float = 0.1
    """E-step KL constraint"""
    mstep_kl_epsilon: float = 0.1
    """M-step KL constraint"""
    per_dim_constraining: bool = False
    """whether to apply per-dimension KL constraints in M-step"""
    lambda_init: float = 1.0
    """initial M-step dual variable value"""
    lambda_lr: float = 3e-4
    """M-step dual optimizer learning rate"""
    action_penalization: bool = False
    """enable optional action penalization term"""
    epsilon_penalty: float = 0.001
    """epsilon used by optional action penalization"""
    max_grad_norm: float = 1.0
    """max gradient norm for clipping"""
    action_samples: int = 256
    """number of sampled actions for MPO E-step"""
    use_retrace: bool = True
    """whether to use Retrace targets"""
    retrace_steps: int = 2
    """sequence length for Retrace"""
    retrace_mc_actions: int = 8
    """MC samples for expected Q in Retrace"""
    retrace_lambda: float = 0.95
    """Retrace lambda"""


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = make_base_env(env_id, seed=seed, render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


@torch.no_grad()
def evaluate(
    agent: MPOAgent,
    env_id: str,
    seed: int,
    n_episodes: int = 10,
) -> dict[str, float]:
    returns = []
    eval_env = make_env(env_id, seed=seed, idx=0, capture_video=False, run_name="eval")()
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
    return {
        "eval/return_mean": float(np.mean(returns_arr)),
        "eval/return_std": float(np.std(returns_arr)),
        "eval/return_min": float(np.min(returns_arr)),
        "eval/return_max": float(np.max(returns_arr)),
    }


def save_checkpoint(agent: MPOAgent, run_name: str, global_step: int) -> str:
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "mpo.pt"
    torch.save(
        {
            "step": int(global_step),
            "policy": agent.policy.state_dict(),
            "policy_target": agent.policy_target.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "q1_target": agent.q1_target.state_dict(),
            "q2_target": agent.q2_target.state_dict(),
            "policy_opt": agent.policy_opt.state_dict(),
            "q_opt": agent.q_opt.state_dict(),
            "dual_opt": agent.dual_opt.state_dict(),
            "log_temperature": agent.log_temperature.detach().cpu(),
            "log_alpha_mean": agent.log_alpha_mean.detach().cpu(),
            "log_alpha_stddev": agent.log_alpha_stddev.detach().cpu(),
            "log_penalty_temperature": (
                None
                if agent.log_penalty_temperature is None
                else agent.log_penalty_temperature.detach().cpu()
            ),
        },
        ckpt_path,
    )
    return str(ckpt_path)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.env_id, seed=args.seed, idx=0, capture_video=args.capture_video, run_name=run_name)()
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("MPO only supports continuous action spaces.")
    if env.action_space.shape is None:
        raise ValueError("Action space has no shape.")

    obs_dim = infer_obs_dim(env.observation_space)
    act_dim = int(np.prod(env.action_space.shape))

    config = MPOConfig(
        gamma=args.gamma,
        tau=args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        kl_epsilon=args.kl_epsilon,
        mstep_kl_epsilon=args.mstep_kl_epsilon,
        per_dim_constraining=args.per_dim_constraining,
        temperature_init=args.temperature_init,
        temperature_lr=args.temperature_lr,
        lambda_init=args.lambda_init,
        lambda_lr=args.lambda_lr,
        action_penalization=args.action_penalization,
        epsilon_penalty=args.epsilon_penalty,
        max_grad_norm=args.max_grad_norm,
        action_samples=args.action_samples,
        use_retrace=args.use_retrace,
        retrace_steps=args.retrace_steps,
        retrace_mc_actions=args.retrace_mc_actions,
        retrace_lambda=args.retrace_lambda,
    )

    agent = MPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        device=device,
        policy_layer_sizes=args.policy_layer_sizes,
        critic_layer_sizes=args.critic_layer_sizes,
        config=config,
    )
    replay = MPOReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=args.buffer_size)

    start_time = time.time()
    obs, _ = env.reset(seed=args.seed)
    obs = flatten_obs(obs)
    episode_return = 0.0
    episode_length = 0

    try:
        for global_step in range(1, args.total_timesteps + 1):
            action_exec, action_raw, behaviour_logp = agent.act_with_logp(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action_exec)
            next_obs = flatten_obs(next_obs)
            reward_f = float(reward)
            done = float(terminated or truncated)

            replay.add(
                obs=obs,
                action_exec=action_exec,
                action_raw=action_raw,
                behaviour_logp=behaviour_logp,
                reward=reward_f,
                next_obs=next_obs,
                done=done,
            )
            obs = next_obs
            episode_return += reward_f
            episode_length += 1

            if terminated or truncated:
                print(f"global_step={global_step}, episodic_return={episode_return}")
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", episode_length, global_step)
                obs, _ = env.reset()
                obs = flatten_obs(obs)
                episode_return = 0.0
                episode_length = 0

            if global_step >= args.update_after and replay.size >= args.batch_size:
                for _ in range(int(args.updates_per_step)):
                    if args.use_retrace and args.retrace_steps > 1:
                        if replay.size < args.batch_size + args.retrace_steps:
                            continue
                        batch = replay.sample_sequences(args.batch_size, seq_len=args.retrace_steps)
                    else:
                        batch = replay.sample(args.batch_size)

                    metrics = agent.update(batch)
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, global_step)

            if global_step % 100 == 0:
                sps = int(global_step / max(1e-8, (time.time() - start_time)))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                eval_metrics = evaluate(
                    agent=agent,
                    env_id=args.env_id,
                    seed=args.seed + 1000,
                    n_episodes=args.eval_episodes,
                )
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, global_step)
                print(
                    f"eval step={global_step} mean={eval_metrics['eval/return_mean']:.3f} "
                    f"std={eval_metrics['eval/return_std']:.3f}"
                )

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                ckpt_path = save_checkpoint(agent, run_name, global_step)
                print(f"saved checkpoint: {ckpt_path}")

        if args.save_model:
            ckpt_path = save_checkpoint(agent, run_name, args.total_timesteps)
            print(f"final model saved: {ckpt_path}")
    finally:
        env.close()
        writer.close()
