import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.evals.mpo_eval import evaluate


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
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel environments (this script currently supports 1)"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    batch_size: int = 256
    """the batch size sampled from replay"""
    updates_per_step: int = 1
    """number of gradient updates per environment step"""
    eval_interval: int = 10_000
    """evaluate every N steps; 0 disables eval"""
    eval_episodes: int = 10
    """number of episodes used for each evaluation"""

    policy_layer_sizes: tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for policy network"""
    critic_layer_sizes: tuple[int, ...] = (256, 256, 256)
    """hidden layer sizes for critic network"""

    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """soft target update factor"""
    policy_lr: float = 3e-4
    """policy optimizer learning rate"""
    q_lr: float = 3e-4
    """critic optimizer learning rate"""

    kl_epsilon: float = 0.1
    """E-step KL constraint"""
    mstep_kl_epsilon: float = 0.1
    """M-step KL constraint"""
    per_dim_constraining: bool = True
    """whether to apply per-dimension KL constraints in M-step"""

    temperature_init: float = 1.0
    """initial E-step dual temperature"""
    temperature_lr: float = 3e-4
    """temperature optimizer learning rate"""
    lambda_init: float = 1.0
    """initial M-step dual variable"""
    lambda_lr: float = 3e-4
    """M-step dual optimizer learning rate"""

    action_samples: int = 20
    """number of target policy sampled actions per next state"""
    max_grad_norm: float = 1.0
    """max gradient norm for clipping"""

    # Optional retrace targets
    use_retrace: bool = False
    """whether to use Retrace targets"""
    retrace_steps: int = 2
    """sequence length for Retrace"""
    retrace_mc_actions: int = 8
    """MC samples for expected Q in Retrace"""
    retrace_lambda: float = 0.95
    """Retrace lambda"""


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


def infer_obs_dim(obs_space: gym.Space) -> int:
    if isinstance(obs_space, gym.spaces.Dict):
        dims = []
        for v in obs_space.spaces.values():
            if v.shape is None:
                raise ValueError("Observation space has no shape.")
            dims.append(int(np.prod(v.shape)))
        return int(sum(dims))
    if obs_space.shape is None:
        raise ValueError("Observation space has no shape.")
    return int(np.prod(obs_space.shape))


def flatten_obs(obs):
    if isinstance(obs, dict):
        if not obs:
            return np.asarray([], dtype=np.float32)

        parts = []
        for key in sorted(obs.keys()):
            p = np.asarray(obs[key], dtype=np.float32)
            if p.ndim == 0:
                p = p.reshape(1)
            parts.append(p)

        leading_dims = [p.shape[0] for p in parts if p.ndim >= 2]
        is_vectorized = len(leading_dims) > 0

        if is_vectorized:
            n = leading_dims[0]
            flat_parts = [
                p.reshape(n, 1) if p.ndim == 1 else p.reshape(n, -1) for p in parts
            ]
            return np.concatenate(flat_parts, axis=1)

        flat_parts = []
        for p in parts:
            if p.ndim != 1:
                raise ValueError(f"Unexpected shape in single-env obs: {p.shape}")
            flat_parts.append(p.reshape(-1))
        return np.concatenate(flat_parts, axis=0)

    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1)


class LayerNormMLP(nn.Module):
    def __init__(
        self, in_dim: int, layer_sizes: Tuple[int, ...], activate_final: bool = False
    ):
        super().__init__()
        if len(layer_sizes) < 1:
            raise ValueError("layer_sizes must have at least one layer")

        layers: list[nn.Module] = [
            nn.Linear(in_dim, layer_sizes[0]),
            nn.LayerNorm(layer_sizes[0]),
            nn.Tanh(),
        ]
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if activate_final or i < len(layer_sizes) - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallInitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, std: float = 0.01):
        super().__init__(in_features, out_features)
        nn.init.trunc_normal_(self.weight, std=std)
        nn.init.zeros_(self.bias)


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_sizes: Tuple[int, ...],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )
        self.encoder = LayerNormMLP(obs_dim + act_dim, layer_sizes, activate_final=True)
        self.head = SmallInitLinear(layer_sizes[-1], 1, std=0.01)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        act = torch.maximum(torch.minimum(act, self.action_high), self.action_low)
        x = torch.cat([obs, act], dim=-1)
        return self.head(self.encoder(x))


class DiagonalGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layer_sizes: Tuple[int, ...],
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ):
        super().__init__()
        self.encoder = LayerNormMLP(obs_dim, layer_sizes, activate_final=True)
        self.policy_mean = nn.Linear(layer_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(layer_sizes[-1], act_dim)

        nn.init.kaiming_normal_(
            self.policy_mean.weight, a=0.0, mode="fan_in", nonlinearity="linear"
        )
        nn.init.zeros_(self.policy_mean.bias)

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        self.register_buffer(
            "action_low", torch.tensor(action_low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(action_high, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mean = self.policy_mean(h)
        log_std = self.policy_logstd(h)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def log_prob(
        self, mean: torch.Tensor, log_std: torch.Tensor, actions_raw: torch.Tensor
    ) -> torch.Tensor:
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = log_std.exp()
        return Normal(mean, std).log_prob(actions_raw).sum(dim=-1, keepdim=True)

    def _clip_to_env_bounds(self, actions_raw: torch.Tensor) -> torch.Tensor:
        return torch.maximum(
            torch.minimum(actions_raw, self.action_high), self.action_low
        )

    def sample_action_raw_and_exec(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_std = torch.clamp(log_std, -20.0, 2.0)
        if deterministic:
            actions_raw = mean
        else:
            std = log_std.exp()
            actions_raw = Normal(mean, std).rsample()
        actions_exec = self._clip_to_env_bounds(actions_raw)
        return actions_raw, actions_exec

    def sample_actions_raw_and_exec(
        self, obs: torch.Tensor, num_actions: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        actions_raw = normal.rsample(sample_shape=(num_actions,)).permute(1, 0, 2)
        actions_exec = self._clip_to_env_bounds(actions_raw)
        return actions_raw, actions_exec

    def sample_actions(self, obs: torch.Tensor, num_actions: int) -> torch.Tensor:
        _, actions_exec = self.sample_actions_raw_and_exec(obs, num_actions)
        return actions_exec


@dataclass
class MPOConfig:
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    kl_epsilon: float = 0.1
    mstep_kl_epsilon: float = 0.1
    per_dim_constraining: bool = True
    temperature_init: float = 1.0
    temperature_lr: float = 3e-4
    lambda_init: float = 1.0
    lambda_lr: float = 3e-4
    max_grad_norm: float = 1.0
    action_samples: int = 20
    use_retrace: bool = False
    retrace_steps: int = 2
    retrace_mc_actions: int = 8
    retrace_lambda: float = 0.95


class MPOReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self._global_step = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_exec = np.zeros((capacity, act_dim), dtype=np.float32)
        self.actions_raw = np.zeros((capacity, act_dim), dtype=np.float32)
        self.behaviour_logp = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.step_ids = np.zeros((capacity,), dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        action_exec: np.ndarray,
        action_raw: np.ndarray,
        behaviour_logp: float,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions_exec[self.ptr] = action_exec
        self.actions_raw[self.ptr] = action_raw
        self.behaviour_logp[self.ptr] = behaviour_logp
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.step_ids[self.ptr] = self._global_step
        self._global_step += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions_exec[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.size < seq_len:
            raise ValueError("Not enough data in replay buffer for sequence sampling")

        obs_dim = self.obs.shape[-1]
        act_dim = self.actions_exec.shape[-1]

        obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        next_obs_b = np.zeros((batch_size, seq_len, obs_dim), dtype=np.float32)
        actions_exec_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        actions_raw_b = np.zeros((batch_size, seq_len, act_dim), dtype=np.float32)
        rewards_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        dones_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        beh_logp_b = np.zeros((batch_size, seq_len, 1), dtype=np.float32)

        if self.size < self.capacity:
            starts = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
            for i, start in enumerate(starts):
                idxs = np.arange(start, start + seq_len)
                obs_b[i] = self.obs[idxs]
                next_obs_b[i] = self.next_obs[idxs]
                actions_exec_b[i] = self.actions_exec[idxs]
                actions_raw_b[i] = self.actions_raw[idxs]
                rewards_b[i] = self.rewards[idxs]
                dones_b[i] = self.dones[idxs]
                beh_logp_b[i] = self.behaviour_logp[idxs]
            return {
                "obs": obs_b,
                "actions_exec": actions_exec_b,
                "actions_raw": actions_raw_b,
                "behaviour_logp": beh_logp_b,
                "rewards": rewards_b,
                "next_obs": next_obs_b,
                "dones": dones_b,
            }

        filled = 0
        max_tries = batch_size * 200
        tries = 0
        while filled < batch_size and tries < max_tries:
            tries += 1
            start = np.random.randint(0, self.capacity)
            idxs = (start + np.arange(seq_len)) % self.capacity
            step_ids = self.step_ids[idxs]
            if not np.all(step_ids[1:] == step_ids[:-1] + 1):
                continue

            obs_b[filled] = self.obs[idxs]
            next_obs_b[filled] = self.next_obs[idxs]
            actions_exec_b[filled] = self.actions_exec[idxs]
            actions_raw_b[filled] = self.actions_raw[idxs]
            rewards_b[filled] = self.rewards[idxs]
            dones_b[filled] = self.dones[idxs]
            beh_logp_b[filled] = self.behaviour_logp[idxs]
            filled += 1

        if filled < batch_size:
            raise RuntimeError(
                "Failed to sample enough contiguous sequences; consider increasing replay size or reducing seq_len."
            )

        return {
            "obs": obs_b,
            "actions_exec": actions_exec_b,
            "actions_raw": actions_raw_b,
            "behaviour_logp": beh_logp_b,
            "rewards": rewards_b,
            "next_obs": next_obs_b,
            "dones": dones_b,
        }


class MPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...],
        critic_layer_sizes: Tuple[int, ...],
        config: MPOConfig,
    ):
        self.device = device
        self.config = config

        self.policy = DiagonalGaussianPolicy(
            obs_dim,
            act_dim,
            layer_sizes=policy_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.policy_target = copy.deepcopy(self.policy).to(device)
        self.policy_target.eval()

        self.q1 = Critic(
            obs_dim,
            act_dim,
            layer_sizes=critic_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.q2 = Critic(
            obs_dim,
            act_dim,
            layer_sizes=critic_layer_sizes,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)
        self.q1_target.eval()
        self.q2_target.eval()

        self.policy_opt = optim.Adam(
            self.policy.parameters(), lr=self.config.policy_lr, eps=1e-5
        )
        self.q_opt = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.config.q_lr,
            eps=1e-5,
        )

        temperature_init_t = torch.tensor(self.config.temperature_init, device=device)
        temperature_init_t = torch.clamp(temperature_init_t, min=1e-8)
        self.log_temperature = nn.Parameter(torch.log(torch.expm1(temperature_init_t)))

        lambda_init_t = torch.tensor(self.config.lambda_init, device=device)
        lambda_init_t = torch.clamp(lambda_init_t, min=1e-8)
        dual_shape = (act_dim,) if self.config.per_dim_constraining else (1,)
        init_log = torch.log(torch.expm1(lambda_init_t)).item()
        self.log_alpha_mean = nn.Parameter(
            torch.full(dual_shape, init_log, device=device)
        )
        self.log_alpha_stddev = nn.Parameter(
            torch.full(dual_shape, init_log, device=device)
        )

        temperature_params = [self.log_temperature]
        alpha_params = [self.log_alpha_mean, self.log_alpha_stddev]
        self.dual_opt = optim.Adam(
            [
                {"params": temperature_params, "lr": self.config.temperature_lr},
                {"params": alpha_params, "lr": self.config.lambda_lr},
            ],
            eps=1e-5,
        )

    def state_dict(self) -> dict[str, dict | torch.Tensor]:
        state = {
            "policy": self.policy.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "log_alpha_mean": self.log_alpha_mean.detach().cpu(),
            "log_alpha_stddev": self.log_alpha_stddev.detach().cpu(),
        }
        return state

    def load_state_dict(self, state_dict: dict[str, dict | torch.Tensor]) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.policy_target.load_state_dict(state_dict["policy_target"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.q1_target.load_state_dict(state_dict["q1_target"])
        self.q2_target.load_state_dict(state_dict["q2_target"])
        self.log_temperature.data.copy_(
            torch.as_tensor(state_dict["log_temperature"], device=self.device)
        )
        self.log_alpha_mean.data.copy_(
            torch.as_tensor(state_dict["log_alpha_mean"], device=self.device)
        )
        self.log_alpha_stddev.data.copy_(
            torch.as_tensor(state_dict["log_alpha_stddev"], device=self.device)
        )

    def _forward_kl_diag_gaussians(
        self,
        mean0: torch.Tensor,
        log_std0: torch.Tensor,
        mean1: torch.Tensor,
        log_std1: torch.Tensor,
    ) -> torch.Tensor:
        var0 = torch.exp(2.0 * log_std0)
        var1 = torch.exp(2.0 * log_std1)
        kl_per_dim = 0.5 * (
            var0 / var1
            + (mean1 - mean0).pow(2) / var1
            - 1.0
            + 2.0 * (log_std1 - log_std0)
        )
        return kl_per_dim.sum(dim=-1, keepdim=True)

    def _kl_diag_gaussian_per_dim(
        self,
        mean_p: torch.Tensor,
        log_std_p: torch.Tensor,
        mean_q: torch.Tensor,
        log_std_q: torch.Tensor,
    ) -> torch.Tensor:
        var_p = torch.exp(2.0 * log_std_p)
        var_q = torch.exp(2.0 * log_std_q)
        return (
            (log_std_q - log_std_p)
            + 0.5 * (var_p + (mean_p - mean_q).pow(2)) / var_q
            - 0.5
        )

    def _compute_weights_and_temperature_loss(
        self,
        q_values: torch.Tensor,
        epsilon: float,
        temperature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_detached = q_values.detach() / temperature
        weights = torch.softmax(q_detached, dim=1).detach()
        q_logsumexp = torch.logsumexp(q_detached, dim=1)
        log_num_actions = math.log(q_values.shape[1])
        loss_temperature = temperature * (
            float(epsilon) + q_logsumexp.mean() - log_num_actions
        )
        return weights, loss_temperature

    def _compute_nonparametric_kl_from_weights(
        self, weights: torch.Tensor
    ) -> torch.Tensor:
        n = float(weights.shape[1])
        integrand = torch.log(n * weights + 1e-8)
        return (weights * integrand).sum(dim=1)

    def act_with_logp(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.policy(obs_t)
            action_raw, action_exec = self.policy.sample_action_raw_and_exec(
                mean, log_std, deterministic
            )
            logp = self.policy.log_prob(mean, log_std, action_raw)
        return (
            action_exec.detach().cpu().numpy().squeeze(0),
            action_raw.detach().cpu().numpy().squeeze(0),
            float(logp.item()),
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action_exec, _, _ = self.act_with_logp(obs, deterministic=deterministic)
        return action_exec

    def _expected_q_current(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            actions = self.policy_target.sample_actions(
                obs, num_actions=self.config.retrace_mc_actions
            )
            batch_size = obs.shape[0]
            obs_rep = obs.unsqueeze(1).expand(
                batch_size, self.config.retrace_mc_actions, obs.shape[-1]
            )
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_flat = actions.reshape(-1, actions.shape[-1])
            q1 = self.q1_target(obs_flat, act_flat)
            q2 = self.q2_target(obs_flat, act_flat)
            q = torch.min(q1, q2)
            return q.reshape(batch_size, self.config.retrace_mc_actions).mean(
                dim=1, keepdim=True
            )

    def _retrace_q_target(self, batch: dict) -> torch.Tensor:
        obs_seq = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_exec_seq = torch.tensor(
            batch["actions_exec"], dtype=torch.float32, device=self.device
        )
        actions_raw_seq = torch.tensor(
            batch["actions_raw"], dtype=torch.float32, device=self.device
        )
        rewards_seq = torch.tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        next_obs_seq = torch.tensor(
            batch["next_obs"], dtype=torch.float32, device=self.device
        )
        dones_seq = torch.tensor(
            batch["dones"], dtype=torch.float32, device=self.device
        )
        behaviour_logp_seq = torch.tensor(
            batch["behaviour_logp"], dtype=torch.float32, device=self.device
        )

        batch_size, seq_len, obs_dim = obs_seq.shape
        act_dim = actions_exec_seq.shape[-1]

        with torch.no_grad():
            obs_flat = obs_seq.reshape(batch_size * seq_len, obs_dim)
            act_exec_flat = actions_exec_seq.reshape(batch_size * seq_len, act_dim)
            q1_t = self.q1_target(obs_flat, act_exec_flat)
            q2_t = self.q2_target(obs_flat, act_exec_flat)
            q_t = torch.min(q1_t, q2_t).reshape(batch_size, seq_len, 1)

            next_obs_flat = next_obs_seq.reshape(batch_size * seq_len, obs_dim)
            v_next = self._expected_q_current(next_obs_flat).reshape(
                batch_size, seq_len, 1
            )

            delta = rewards_seq + (1.0 - dones_seq) * self.config.gamma * v_next - q_t

            mean, log_std = self.policy_target(obs_flat)
            actions_raw_flat = actions_raw_seq.reshape(batch_size * seq_len, act_dim)
            log_pi = self.policy_target.log_prob(
                mean, log_std, actions_raw_flat
            ).reshape(batch_size, seq_len, 1)
            log_ratio = log_pi - behaviour_logp_seq
            rho = torch.exp(log_ratio).squeeze(-1)
            c = self.config.retrace_lambda * torch.minimum(torch.ones_like(rho), rho)

            q_ret = q_t[:, 0, :].clone()
            cont = torch.ones((batch_size, 1), device=self.device)
            c_prod = torch.ones((batch_size, 1), device=self.device)
            discount = torch.ones((batch_size, 1), device=self.device)
            dones_flat = dones_seq.squeeze(-1)

            for t in range(seq_len):
                if t > 0:
                    cont = cont * (1.0 - dones_flat[:, t - 1 : t])
                    c_prod = c_prod * c[:, t : t + 1]
                    discount = discount * self.config.gamma
                q_ret = q_ret + cont * discount * c_prod * delta[:, t, :]

        return q_ret

    def update(self, batch: dict) -> dict:
        is_sequence_batch = (
            isinstance(batch.get("obs"), np.ndarray) and batch["obs"].ndim == 3
        )

        if (
            self.config.use_retrace
            and is_sequence_batch
            and self.config.retrace_steps > 1
        ):
            target = self._retrace_q_target(batch)
            obs = torch.tensor(
                batch["obs"][:, 0, :], dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                batch["actions_exec"][:, 0, :], dtype=torch.float32, device=self.device
            )
        else:
            obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
            actions = torch.tensor(
                batch["actions"], dtype=torch.float32, device=self.device
            )
            rewards = torch.tensor(
                batch["rewards"], dtype=torch.float32, device=self.device
            )
            next_obs = torch.tensor(
                batch["next_obs"], dtype=torch.float32, device=self.device
            )
            dones = torch.tensor(
                batch["dones"], dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                next_actions = self.policy_target.sample_actions(
                    next_obs, num_actions=self.config.action_samples
                )
                batch_size = next_obs.shape[0]
                next_obs_rep = next_obs.unsqueeze(1).expand(
                    batch_size, self.config.action_samples, next_obs.shape[-1]
                )
                next_obs_flat = next_obs_rep.reshape(-1, next_obs.shape[-1])
                next_act_flat = next_actions.reshape(-1, next_actions.shape[-1])
                q1_target = self.q1_target(next_obs_flat, next_act_flat)
                q2_target = self.q2_target(next_obs_flat, next_act_flat)
                q_target = (
                    torch.min(q1_target, q2_target)
                    .reshape(batch_size, self.config.action_samples)
                    .mean(dim=1, keepdim=True)
                )
                target = rewards + (1.0 - dones) * self.config.gamma * q_target

        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)

        q_loss = q1_loss + q2_loss
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.config.max_grad_norm,
        )
        self.q_opt.step()

        batch_size = obs.shape[0]
        num_samples = self.config.action_samples

        mean_online, log_std_online = self.policy(obs)
        with torch.no_grad():
            mean_target, log_std_target = self.policy_target(obs)
            sampled_actions_raw, sampled_actions_exec = (
                self.policy_target.sample_actions_raw_and_exec(
                    obs, num_actions=num_samples
                )
            )
            obs_rep = obs.unsqueeze(1).expand(batch_size, num_samples, obs.shape[-1])
            obs_flat = obs_rep.reshape(-1, obs.shape[-1])
            act_exec_flat = sampled_actions_exec.reshape(
                -1, sampled_actions_exec.shape[-1]
            )
            q1_vals = self.q1_target(obs_flat, act_exec_flat)
            q2_vals = self.q2_target(obs_flat, act_exec_flat)
            q_vals = torch.min(q1_vals, q2_vals).reshape(batch_size, num_samples)

        temperature = F.softplus(self.log_temperature) + 1e-8
        weights, loss_temperature = self._compute_weights_and_temperature_loss(
            q_vals, self.config.kl_epsilon, temperature
        )

        penalty_kl_rel = torch.tensor(0.0, device=self.device)

        kl_nonparametric = self._compute_nonparametric_kl_from_weights(weights)
        kl_q_rel = kl_nonparametric.mean() / float(self.config.kl_epsilon)

        std_online = torch.exp(log_std_online)
        std_target = torch.exp(log_std_target)

        actions_sampled = sampled_actions_raw.detach()
        mean_online_exp = mean_online.unsqueeze(1)
        std_online_exp = std_online.unsqueeze(1)
        mean_target_exp = mean_target.unsqueeze(1)
        std_target_exp = std_target.unsqueeze(1)

        log_prob_fixed_stddev = (
            Normal(mean_online_exp, std_target_exp)
            .log_prob(actions_sampled)
            .sum(dim=-1)
        )
        log_prob_fixed_mean = (
            Normal(mean_target_exp, std_online_exp)
            .log_prob(actions_sampled)
            .sum(dim=-1)
        )

        loss_policy_mean = -(weights * log_prob_fixed_stddev).sum(dim=1).mean()
        loss_policy_std = -(weights * log_prob_fixed_mean).sum(dim=1).mean()
        loss_policy = loss_policy_mean + loss_policy_std

        if self.config.per_dim_constraining:
            kl_mean = self._kl_diag_gaussian_per_dim(
                mean_target.detach(),
                log_std_target.detach(),
                mean_online,
                log_std_target.detach(),
            )
            kl_std = self._kl_diag_gaussian_per_dim(
                mean_target.detach(),
                log_std_target.detach(),
                mean_target.detach(),
                log_std_online,
            )
        else:
            kl_mean = self._forward_kl_diag_gaussians(
                mean_target.detach(),
                log_std_target.detach(),
                mean_online,
                log_std_target.detach(),
            )
            kl_std = self._forward_kl_diag_gaussians(
                mean_target.detach(),
                log_std_target.detach(),
                mean_target.detach(),
                log_std_online,
            )

        mean_kl_mean = kl_mean.mean(dim=0)
        mean_kl_std = kl_std.mean(dim=0)

        alpha_mean = F.softplus(self.log_alpha_mean) + 1e-8
        alpha_std = F.softplus(self.log_alpha_stddev) + 1e-8

        loss_kl_mean = (alpha_mean.detach() * mean_kl_mean).sum()
        loss_kl_std = (alpha_std.detach() * mean_kl_std).sum()
        loss_kl_penalty = loss_kl_mean + loss_kl_std

        loss_alpha_mean = (
            alpha_mean * (self.config.mstep_kl_epsilon - mean_kl_mean.detach())
        ).sum()
        loss_alpha_std = (
            alpha_std * (self.config.mstep_kl_epsilon - mean_kl_std.detach())
        ).sum()

        dual_loss = loss_temperature + loss_alpha_mean + loss_alpha_std
        self.dual_opt.zero_grad()
        dual_loss.backward()
        dual_params = [
            p
            for p in [self.log_temperature, self.log_alpha_mean, self.log_alpha_stddev]
            if p is not None
        ]
        nn.utils.clip_grad_norm_(dual_params, self.config.max_grad_norm)
        self.dual_opt.step()

        policy_total_loss = loss_policy + loss_kl_penalty
        self.policy_opt.zero_grad()
        policy_total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_opt.step()

        self._soft_update_module(self.q1, self.q1_target)
        self._soft_update_module(self.q2, self.q2_target)
        self._soft_update_module(self.policy, self.policy_target)

        temperature_val = float(
            (F.softplus(self.log_temperature) + 1e-8).detach().item()
        )
        lambda_val = float(
            (F.softplus(self.log_alpha_mean).mean() + 1e-8).detach().item()
        )

        return {
            "loss/q1": float(q1_loss.item()),
            "loss/q2": float(q2_loss.item()),
            "loss/policy": float(loss_policy.item()),
            "loss/dual_eta": float(loss_temperature.detach().item()),
            "loss/dual": float(dual_loss.detach().item()),
            "kl/q_pi": float(kl_q_rel.detach().item()),
            "kl/mean": float(mean_kl_mean.mean().detach().item()),
            "kl/std": float(mean_kl_std.mean().detach().item()),
            "eta": temperature_val,
            "lambda": lambda_val,
            "alpha_mean": float(
                (F.softplus(self.log_alpha_mean) + 1e-8).mean().detach().item()
            ),
            "alpha_std": float(
                (F.softplus(self.log_alpha_stddev) + 1e-8).mean().detach().item()
            ),
            "q/min": float(q_vals.min().detach().item()),
            "q/max": float(q_vals.max().detach().item()),
            "pi/std_min": float(std_online.min().detach().item()),
            "pi/std_max": float(std_online.max().detach().item()),
            "penalty_kl/q_pi": float(penalty_kl_rel.detach().item()),
        }

    def _soft_update_module(self, net: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for param, target_param in zip(net.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.num_envs != 1:
        raise ValueError(
            "mpo_continuous_action.py currently supports --num-envs 1 only"
        )

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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(
        args.env_id,
        idx=0,
        capture_video=args.capture_video,
        run_name=run_name,
        gamma=args.gamma
    )()
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("MPO only supports continuous action spaces")
    if env.action_space.shape is None:
        raise ValueError("Action space has no shape")

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
    replay = MPOReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, capacity=args.buffer_size
    )

    obs, _ = env.reset(seed=args.seed)
    obs = flatten_obs(obs)
    episode_return = 0.0
    episode_length = 0
    train_start_time: float | None = None
    update_step = 0

    try:
        for global_step in range(1, args.total_timesteps + 1):
            if train_start_time is None and global_step >= args.learning_starts:
                train_start_time = time.time()

            if global_step < args.learning_starts:
                action_exec = env.action_space.sample().astype(np.float32)
                action_raw = np.copy(action_exec)
                behaviour_logp = 0.0
            else:
                action_exec, action_raw, behaviour_logp = agent.act_with_logp(
                    obs, deterministic=False
                )

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

            if global_step >= args.learning_starts and replay.size >= args.batch_size:
                for _ in range(int(args.updates_per_step)):
                    if args.use_retrace and args.retrace_steps > 1:
                        if replay.size < args.batch_size + args.retrace_steps:
                            continue
                        batch = replay.sample_sequences(
                            args.batch_size, seq_len=args.retrace_steps
                        )
                    else:
                        batch = replay.sample(args.batch_size)

                    metrics = agent.update(batch)
                    update_step += 1
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, update_step)

            if (
                global_step >= args.learning_starts
                and global_step % 100 == 0
                and train_start_time is not None
            ):
                train_env_steps = global_step - args.learning_starts
                sps = int(train_env_steps / max(1e-8, time.time() - train_start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

            if (
                args.eval_interval > 0
                and global_step >= args.learning_starts
                and global_step % args.eval_interval == 0
            ):
                eval_metrics = evaluate(
                    agent=agent,
                    make_env=make_env,
                    flatten_obs=flatten_obs,
                    env_id=args.env_id,
                    seed=args.seed + 1000,
                    n_episodes=args.eval_episodes,
                )
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, global_step)
                print(
                    f"eval step={global_step} "
                    f"mean={eval_metrics['eval/return_mean']:.3f} "
                    f"std={eval_metrics['eval/return_std']:.3f}"
                )

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

            _, episodic_returns = evaluate(
                agent=agent,
                make_env=make_env,
                flatten_obs=flatten_obs,
                env_id=args.env_id,
                seed=args.seed + 1000,
                n_episodes=10,
                run_name=f"{run_name}-eval",
                capture_video=True,
                return_episode_returns=True,
            )
            for idx, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, idx)

            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub

                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = (
                    f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                )
                push_to_hub(
                    args,
                    episodic_returns,
                    repo_id,
                    "MPO",
                    f"runs/{run_name}",
                    f"videos/{run_name}-eval",
                )

    finally:
        env.close()
        writer.close()
