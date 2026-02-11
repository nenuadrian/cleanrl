# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
# This is a PPO continuous-control variant refactored into a VMPO-style trainer layout:
# - class-based trainer
# - rollout buffers as lists
# - step-driven loop with done-mask episode logging
# PPO optimization logic is kept equivalent to cleanrl/ppo_continuous_action.py.
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


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
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of PPO updates (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PPOTrainer:
    def __init__(self, args: Args, device: torch.device, writer: SummaryWriter, run_name: str):
        self.args = args
        self.device = device
        self.writer = writer
        self.run_name = run_name

        env_fns = [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        self.envs = gym.vector.AsyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        self.agent = Agent(self.envs).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

        self.global_step = 0
        self.update_idx = 0
        self.start_time = time.time()

        next_obs, _ = self.envs.reset(seed=args.seed)
        self.next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        self.episode_return = np.zeros(args.num_envs, dtype=np.float32)
        self.episode_length = np.zeros(args.num_envs, dtype=np.int32)

        self._reset_rollout()

    def _reset_rollout(self):
        self.obs_buf: list[torch.Tensor] = []
        self.actions_buf: list[torch.Tensor] = []
        self.logprobs_buf: list[torch.Tensor] = []
        self.rewards_buf: list[torch.Tensor] = []
        self.dones_buf: list[torch.Tensor] = []
        self.values_buf: list[torch.Tensor] = []

    def _rollout_full(self) -> bool:
        return len(self.obs_buf) >= self.args.num_steps

    def _anneal_lr(self):
        if not self.args.anneal_lr:
            return
        frac = 1.0 - (self.update_idx - 1.0) / max(1, self.args.num_iterations)
        lr_now = frac * self.args.learning_rate
        self.optimizer.param_groups[0]["lr"] = lr_now

    def _collect_step(self):
        obs_t = self.next_obs

        with torch.no_grad():
            action_t, logprob_t, _, value_t = self.agent.get_action_and_value(obs_t)

        next_obs, reward, terminated, truncated, _ = self.envs.step(action_t.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        reward = np.asarray(reward, dtype=np.float32)

        self.obs_buf.append(obs_t)
        self.actions_buf.append(action_t)
        self.logprobs_buf.append(logprob_t)
        self.values_buf.append(value_t.flatten())
        self.rewards_buf.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.dones_buf.append(torch.tensor(done, dtype=torch.float32, device=self.device))

        self.next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        self.global_step += self.args.num_envs

        self.episode_return += reward
        self.episode_length += 1
        finished_mask = done.astype(bool)
        if np.any(finished_mask):
            finished_returns = self.episode_return[finished_mask]
            finished_lengths = self.episode_length[finished_mask]
            self.episode_return[finished_mask] = 0.0
            self.episode_length[finished_mask] = 0

            self.writer.add_scalar("charts/episodic_return_mean", float(np.mean(finished_returns)), self.global_step)
            self.writer.add_scalar("charts/episodic_length_mean", float(np.mean(finished_lengths)), self.global_step)
            for ep_return, ep_len in zip(finished_returns, finished_lengths):
                self.writer.add_scalar("charts/episodic_return", float(ep_return), self.global_step)
                self.writer.add_scalar("charts/episodic_length", float(ep_len), self.global_step)

    def _compute_advantages_returns(self):
        rewards = torch.stack(self.rewards_buf)  # (T, N)
        dones = torch.stack(self.dones_buf)  # (T, N)
        values = torch.stack(self.values_buf)  # (T, N)

        with torch.no_grad():
            next_value = self.agent.get_value(self.next_obs).flatten()  # (N,)

        advantages = torch.zeros_like(rewards, device=self.device)
        lastgaelam = torch.zeros(self.args.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_values = next_value
            else:
                next_values = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.args.gamma * next_values * next_nonterminal - values[t]
            lastgaelam = delta + self.args.gamma * self.args.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        obs = torch.stack(self.obs_buf)
        actions = torch.stack(self.actions_buf)
        logprobs = torch.stack(self.logprobs_buf)
        return obs, actions, logprobs, values, advantages, returns

    def _update_from_rollout(self):
        self.update_idx += 1
        self._anneal_lr()

        obs, actions, logprobs, values, advantages, returns = self._compute_advantages_returns()

        b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(b_obs.shape[0])
        clipfracs = []
        old_approx_kl = torch.tensor(0.0, device=self.device)
        approx_kl = torch.tensor(0.0, device=self.device)
        pg_loss = torch.tensor(0.0, device=self.device)
        v_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, b_obs.shape[0], self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl.item() > self.args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "entropy": float(entropy_loss.item()),
            "old_approx_kl": float(old_approx_kl.item()),
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": float(explained_var),
        }

    def train(self):
        while self.global_step < self.args.total_timesteps:
            self._collect_step()

            if not self._rollout_full():
                continue

            metrics = self._update_from_rollout()
            self.writer.add_scalar("charts/learning_rate", metrics["learning_rate"], self.global_step)
            self.writer.add_scalar("losses/value_loss", metrics["value_loss"], self.global_step)
            self.writer.add_scalar("losses/policy_loss", metrics["policy_loss"], self.global_step)
            self.writer.add_scalar("losses/entropy", metrics["entropy"], self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", metrics["old_approx_kl"], self.global_step)
            self.writer.add_scalar("losses/approx_kl", metrics["approx_kl"], self.global_step)
            self.writer.add_scalar("losses/clipfrac", metrics["clipfrac"], self.global_step)
            self.writer.add_scalar("losses/explained_variance", metrics["explained_variance"], self.global_step)

            sps = int(self.global_step / (time.time() - self.start_time))
            print("SPS:", sps)
            self.writer.add_scalar("charts/SPS", sps, self.global_step)

            self._reset_rollout()

    def save_and_eval(self):
        model_path = f"runs/{self.run_name}/{self.args.exp_name}.cleanrl_model"
        torch.save(self.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            self.args.env_id,
            eval_episodes=10,
            run_name=f"{self.run_name}-eval",
            Model=Agent,
            device=self.device,
            gamma=self.args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            self.writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if self.args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{self.args.env_id}-{self.args.exp_name}-seed{self.args.seed}"
            repo_id = f"{self.args.hf_entity}/{repo_name}" if self.args.hf_entity else repo_name
            push_to_hub(
                self.args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{self.run_name}",
                f"videos/{self.run_name}-eval",
            )

    def close(self):
        self.envs.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by num_minibatches ({args.num_minibatches})"
        )
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int((args.total_timesteps + args.batch_size - 1) // args.batch_size)
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

    trainer = PPOTrainer(args=args, device=device, writer=writer, run_name=run_name)
    try:
        trainer.train()
        if args.save_model:
            trainer.save_and_eval()
    finally:
        trainer.close()
        writer.close()
