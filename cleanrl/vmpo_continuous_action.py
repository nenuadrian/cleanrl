import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """the id of the environment"""
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per rollout"""
    updates_per_rollout: int = 3
    """number of gradient updates after each rollout"""
    eval_interval: int = 0
    """evaluate every N env steps; 0 disables evaluation"""
    eval_episodes: int = 100
    """number of episodes per evaluation"""

    policy_layer_sizes: tuple[int, ...] = (256, 256)
    """hidden layer sizes for the policy network"""
    value_layer_sizes: tuple[int, ...] = (512, 256)
    """hidden layer sizes for the value network"""

    gamma: float = 0.99
    """discount factor"""
    advantage_estimator: Literal["returns", "dae", "gae"] = "returns"
    """advantage estimator: `returns` (MC baseline), `dae` (direct TD advantage), or `gae`"""
    gae_lambda: float = 0.95
    """lambda used by GAE when `advantage_estimator=gae`"""
    topk_fraction: float = 0.3
    """fraction of highest-advantage samples used in E-step"""
    temperature_init: float = 2.0
    """initial value for VMPO temperature dual variable"""
    temperature_lr: float = 1e-4
    """learning rate for temperature dual optimizer"""
    epsilon_eta: float = 0.05
    """VMPO epsilon_eta dual constraint"""
    epsilon_mu: float = 0.01
    """VMPO epsilon_mu trust region constraint"""
    epsilon_sigma: float = 0.01
    """VMPO epsilon_sigma trust region constraint"""
    alpha_lr: float = 1e-4
    """learning rate for alpha dual optimizer"""
    policy_lr: float = 3e-4
    """policy network learning rate"""
    value_lr: float = 1e-3
    """value network learning rate"""
    optimizer: Literal["adam", "sgd"] = "adam"
    """optimizer used for policy/value and dual updates"""
    sgd_momentum: float = 0.9
    """momentum when `optimizer=sgd`"""
    max_grad_norm: float = 0.5
    """maximum gradient clipping norm"""

def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
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


class VMPOEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        layer_sizes: Tuple[int, ...],
        activate_final: bool = True,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, layer_sizes[0]))
        layers.append(nn.LayerNorm(layer_sizes[0]))
        layers.append(nn.Tanh())

        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            if activate_final or i < len(layer_sizes) - 1:
                layers.append(nn.ELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        policy_layer_sizes: Tuple[int, ...],
        value_layer_sizes: Tuple[int, ...],
        action_low: np.ndarray | None,
        action_high: np.ndarray | None,
        shared_encoder: bool = False,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        if shared_encoder:
            self.policy_encoder = VMPOEncoder(obs_dim, policy_layer_sizes)
            self.value_encoder = self.policy_encoder
        else:
            self.policy_encoder = VMPOEncoder(obs_dim, policy_layer_sizes)
            self.value_encoder = VMPOEncoder(obs_dim, value_layer_sizes)

        self.policy_mean = nn.Linear(policy_layer_sizes[-1], act_dim)
        self.policy_logstd = nn.Linear(policy_layer_sizes[-1], act_dim)
        self.value_head = nn.Linear(value_layer_sizes[-1], 1)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        if action_low is None or action_high is None:
            action_low = -np.ones(act_dim, dtype=np.float32)
            action_high = np.ones(act_dim, dtype=np.float32)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

        nn.init.xavier_uniform_(self.policy_mean.weight)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.xavier_uniform_(self.policy_logstd.weight)
        nn.init.constant_(self.policy_logstd.bias, -0.5)

    def get_policy_dist_params(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.policy_encoder(obs)
        mean = self.policy_mean(h)
        log_std = self.policy_logstd(h)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_policy_dist_params(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.value_encoder(obs)
        v_hat = self.value_head(h)
        return v_hat

    def forward_all(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.get_policy_dist_params(obs)
        h_val = (
            self.policy_encoder(obs) if self.shared_encoder else self.value_encoder(obs)
        )
        v_hat = self.value_head(h_val)
        v = v_hat
        return mean, log_std, v

    def log_prob(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        std = log_std.exp()
        normal = Normal(mean, std)

        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -0.999999, 0.999999)
        x_t = 0.5 * torch.log((1.0 + y_t) / (1.0 - y_t))

        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(self.action_scale).sum()
        return log_prob

    def sample_action(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            y_t = torch.tanh(mean)
            log_prob = torch.zeros((mean.shape[0], 1), device=mean.device)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob = log_prob - torch.log(self.action_scale).sum()

        action = y_t * self.action_scale + self.action_bias
        return action, log_prob


@dataclass
class VMPOConfig:
    gamma: float = 0.99
    policy_lr: float = 5e-4
    value_lr: float = 1e-3
    optimizer: Literal["adam", "sgd"] = "adam"
    sgd_momentum: float = 0.9
    topk_fraction: float = 0.5
    temperature_init: float = 1.0
    temperature_lr: float = 1e-4
    epsilon_eta: float = 0.1
    epsilon_mu: float = 0.01
    epsilon_sigma: float = 0.01
    alpha_lr: float = 1e-4
    max_grad_norm: float = 10.0


class VMPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: VMPOConfig,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        policy_layer_sizes: Tuple[int, ...] = (256, 256),
        value_layer_sizes: Tuple[int, ...] = (256, 256),
    ):
        self.device = device
        self.config = config

        self.policy = SquashedGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            policy_layer_sizes=policy_layer_sizes,
            value_layer_sizes=value_layer_sizes,
            action_low=action_low,
            action_high=action_high,
            shared_encoder=False,
        ).to(device)

        self.opt = self._build_optimizer(
            [
                {
                    "params": self.policy.policy_encoder.parameters(),
                    "lr": self.config.policy_lr,
                },
                {
                    "params": self.policy.policy_mean.parameters(),
                    "lr": self.config.policy_lr,
                },
                {
                    "params": self.policy.policy_logstd.parameters(),
                    "lr": self.config.policy_lr,
                },
                {
                    "params": self.policy.value_encoder.parameters(),
                    "lr": self.config.value_lr,
                },
                {
                    "params": self.policy.value_head.parameters(),
                    "lr": self.config.value_lr,
                },
            ],
        )

        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(self.config.temperature_init, device=device))
        )
        self.eta_opt = self._build_optimizer(
            [self.log_temperature],
            lr=self.config.temperature_lr,
            eps=1e-5,
        )

        self.log_alpha_mu = nn.Parameter(torch.tensor(np.log(1.0), device=device))
        self.log_alpha_sigma = nn.Parameter(torch.tensor(np.log(1.0), device=device))
        self.alpha_opt = self._build_optimizer(
            [self.log_alpha_mu, self.log_alpha_sigma],
            lr=self.config.alpha_lr,
            eps=1e-5,
        )

    def _build_optimizer(
        self, params: Any, lr: float | None = None, eps: float = 1e-8
    ) -> torch.optim.Optimizer:
        if self.config.optimizer == "adam":
            kwargs: Dict[str, Any] = {"eps": eps}
            if lr is not None:
                kwargs["lr"] = lr
            return torch.optim.Adam(params, **kwargs)
        if self.config.optimizer == "sgd":
            kwargs = {"momentum": self.config.sgd_momentum}
            if lr is not None:
                kwargs["lr"] = lr
            return torch.optim.SGD(params, **kwargs)
        raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "log_alpha_mu": self.log_alpha_mu.detach().cpu(),
            "log_alpha_sigma": self.log_alpha_sigma.detach().cpu(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dict["policy"])
        self.log_temperature.data.copy_(
            torch.as_tensor(state_dict["log_temperature"], device=self.device)
        )
        self.log_alpha_mu.data.copy_(
            torch.as_tensor(state_dict["log_alpha_mu"], device=self.device)
        )
        self.log_alpha_sigma.data.copy_(
            torch.as_tensor(state_dict["log_alpha_sigma"], device=self.device)
        )

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs_np = np.asarray(obs)
        is_batch = obs_np.ndim > 1
        if not is_batch:
            obs_np = obs_np[None, ...]

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, log_std, value = self.policy.forward_all(obs_t)
            action_t, _ = self.policy.sample_action(mean, log_std, deterministic)

        action_np = action_t.cpu().numpy()
        value_np = value.cpu().numpy().squeeze(-1)
        mean_np = mean.cpu().numpy()
        log_std_np = log_std.cpu().numpy()

        if not is_batch:
            return (
                action_np[0],
                np.array([value_np.item()], dtype=np.float32),
                mean_np[0],
                log_std_np[0],
            )
        return action_np, value_np, mean_np, log_std_np

    def value(self, obs: np.ndarray) -> np.ndarray:
        obs_np = np.asarray(obs)
        is_batch = obs_np.ndim > 1
        if not is_batch:
            obs_np = obs_np[None, ...]

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.policy.get_value(obs_t).squeeze(-1)

        v_np = v.cpu().numpy()
        if not is_batch:
            return np.array([v_np.item()], dtype=np.float32)
        return v_np

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        old_means = batch["old_means"]
        old_log_stds = batch["old_log_stds"]

        returns_raw = batch["returns"].squeeze(-1)
        advantages = batch["advantages"].squeeze(-1)

        with torch.no_grad():
            params_before = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()

        with torch.no_grad():
            k = max(1, int(self.config.topk_fraction * advantages.numel()))
            topk_vals, topk_idx = torch.topk(advantages, k, sorted=False)
            threshold = topk_vals.min()
            mask_bool = torch.zeros_like(advantages, dtype=torch.bool)
            mask_bool[topk_idx] = True
            A_sel = advantages[topk_idx]
            K_scalar = float(k)

        eta = F.softplus(self.log_temperature) + 1e-8
        A_max = A_sel.max().detach()
        log_mean_exp = (
            torch.logsumexp((A_sel - A_max) / eta, dim=0)
            - np.log(K_scalar)
            + (A_max / eta)
        )
        dual_loss = eta * self.config.epsilon_eta + eta * log_mean_exp

        self.eta_opt.zero_grad()
        dual_loss.backward()
        self.eta_opt.step()

        with torch.no_grad():
            eta_final = F.softplus(self.log_temperature) + 1e-8
            log_weights = A_sel / eta_final
            weights = torch.softmax(log_weights, dim=0)
            ess = 1.0 / (weights.pow(2).sum() + 1e-12)
            selected_frac = float(K_scalar) / float(advantages.numel())
            adv_std_over_temperature = (advantages.std() / (eta_final + 1e-12)).item()

        current_mean, current_log_std = self.policy(obs)

        log_prob = self.policy.log_prob(current_mean, current_log_std, actions).squeeze(
            -1
        )
        log_prob_sel = log_prob[mask_bool]
        weighted_nll = -(weights.detach() * log_prob_sel).sum()

        with torch.no_grad():
            old_std = old_log_stds.exp()
            new_std = current_log_std.exp()
            kl_mean_all = (
                ((current_mean - old_means) ** 2 / (2.0 * old_std**2 + 1e-8))
                .sum(dim=-1)
                .mean()
            )
            kl_std_all = (
                (
                    (current_log_std - old_log_stds)
                    + (old_std**2) / (2.0 * (new_std**2 + 1e-8))
                    - 0.5
                )
                .sum(dim=-1)
                .mean()
            )

        mean_sel = current_mean[mask_bool]
        log_std_sel = current_log_std[mask_bool]
        old_mean_sel = old_means[mask_bool]
        old_log_std_sel = old_log_stds[mask_bool]

        old_std_sel = old_log_std_sel.exp()
        new_std_sel = log_std_sel.exp()

        kl_mu_sel = (
            (0.5 * ((mean_sel - old_mean_sel) ** 2 / (old_std_sel**2 + 1e-8)))
            .sum(dim=-1)
            .mean()
        )
        kl_sigma_sel = (
            (
                (log_std_sel - old_log_std_sel)
                + (old_std_sel**2) / (2.0 * (new_std_sel**2 + 1e-8))
                - 0.5
            )
            .sum(dim=-1)
            .mean()
        )

        alpha_mu = F.softplus(self.log_alpha_mu) + 1e-8
        alpha_sigma = F.softplus(self.log_alpha_sigma) + 1e-8

        alpha_loss = alpha_mu * (
            self.config.epsilon_mu - kl_mu_sel.detach()
        ) + alpha_sigma * (self.config.epsilon_sigma - kl_sigma_sel.detach())

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            alpha_mu_det = F.softplus(self.log_alpha_mu).detach() + 1e-8
            alpha_sigma_det = F.softplus(self.log_alpha_sigma).detach() + 1e-8

        policy_loss = (
            weighted_nll + (alpha_mu_det * kl_mu_sel) + (alpha_sigma_det * kl_sigma_sel)
        )

        v_hat = self.policy.get_value(obs).squeeze(-1)
        target_hat = returns_raw
        value_loss = 0.5 * F.mse_loss(v_hat, target_hat.detach())

        total_loss = policy_loss + value_loss

        self.opt.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.opt.step()

        with torch.no_grad():
            params_after = nn.utils.parameters_to_vector(
                self.policy.parameters()
            ).detach()
            param_delta = torch.norm(params_after - params_before).item()

            v_pred = self.policy.get_value(obs).squeeze(-1)
            y = returns_raw
            var_y = y.var(unbiased=False)
            explained_var = 1.0 - (y - v_pred).var(unbiased=False) / (var_y + 1e-8)

            mean_eval, log_std_eval = self.policy(obs)
            action_eval, _ = self.policy.sample_action(
                mean_eval, log_std_eval, deterministic=True
            )
            mean_abs_action = float(action_eval.abs().mean().item())

            entropy = (
                (0.5 * (1 + torch.log(2 * torch.pi * new_std_sel**2))).sum(-1).mean()
            )

        metrics = {
            "loss/total": float(total_loss.item()),
            "loss/policy": float(policy_loss.item()),
            "loss/policy_weighted_nll": float(weighted_nll.item()),
            "loss/policy_kl_mean_pen": float((alpha_mu_det * kl_mu_sel).item()),
            "loss/policy_kl_std_pen": float((alpha_sigma_det * kl_sigma_sel).item()),
            "loss/alpha": float(alpha_loss.item()),
            "loss/value": float(value_loss.item()),
            "kl/mean": float(kl_mean_all.item()),
            "kl/std": float(kl_std_all.item()),
            "kl/mean_sel": float(kl_mu_sel.item()),
            "kl/std_sel": float(kl_sigma_sel.item()),
            "vmpo/alpha_mu": float(alpha_mu_det.item()),
            "vmpo/alpha_sigma": float(alpha_sigma_det.item()),
            "vmpo/dual_loss": float(dual_loss.item()),
            "vmpo/epsilon_eta": float(self.config.epsilon_eta),
            "vmpo/temperature_raw": float(eta_final.item()),
            "vmpo/adv_std_over_temperature": float(adv_std_over_temperature),
            "vmpo/selected_frac": float(selected_frac),
            "vmpo/threshold": float(threshold.item()),
            "vmpo/ess": float(ess.item()),
            "train/entropy": float(entropy.item()),
            "train/param_delta": float(param_delta),
            "train/mean_abs_action": float(mean_abs_action),
            "grad/norm": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
            "adv/raw_mean": float(advantages.mean().item()),
            "adv/raw_std": float((advantages.std(unbiased=False) + 1e-8).item()),
            "returns/raw_mean": float(returns_raw.mean().item()),
            "returns/raw_std": float((returns_raw.std(unbiased=False) + 1e-8).item()),
            "value/explained_var": float(explained_var.item()),
        }
        return metrics


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
) -> np.ndarray:
    rewards_np = np.asarray(rewards)
    dones_np = np.asarray(dones)

    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)

    T, N = rewards_np.shape
    returns = np.zeros_like(rewards_np, dtype=np.float32)

    R = np.zeros(N, dtype=np.float32)
    last_val_arr = np.asarray(last_value)
    if last_val_arr.ndim == 0:
        R[:] = float(last_val_arr)
    else:
        R[:] = last_val_arr

    for t in reversed(range(T)):
        R = rewards_np[t] + gamma * (1.0 - dones_np[t]) * R
        returns[t] = R

    if returns.shape[1] == 1:
        return returns.reshape(-1)
    return returns


def compute_dae_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)

    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)
    if values_np.ndim == 1:
        values_np = values_np.reshape(-1, 1)

    T, N = rewards_np.shape
    last_val_arr = np.asarray(last_value, dtype=np.float32)
    if last_val_arr.ndim == 0:
        last_val_arr = np.full((N,), float(last_val_arr), dtype=np.float32)
    else:
        last_val_arr = last_val_arr.reshape(N)

    next_values = np.zeros((T, N), dtype=np.float32)
    if T > 1:
        next_values[:-1] = values_np[1:]
    next_values[-1] = last_val_arr

    returns = rewards_np + gamma * (1.0 - dones_np) * next_values
    advantages = returns - values_np

    if returns.shape[1] == 1:
        return returns.reshape(-1), advantages.reshape(-1)
    return returns, advantages


def compute_gae_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_np = np.asarray(rewards, dtype=np.float32)
    dones_np = np.asarray(dones, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)

    if rewards_np.ndim == 1:
        rewards_np = rewards_np.reshape(-1, 1)
    if dones_np.ndim == 1:
        dones_np = dones_np.reshape(-1, 1)
    if values_np.ndim == 1:
        values_np = values_np.reshape(-1, 1)

    T, N = rewards_np.shape
    last_val_arr = np.asarray(last_value, dtype=np.float32)
    if last_val_arr.ndim == 0:
        last_val_arr = np.full((N,), float(last_val_arr), dtype=np.float32)
    else:
        last_val_arr = last_val_arr.reshape(N)

    next_values = np.zeros((T, N), dtype=np.float32)
    if T > 1:
        next_values[:-1] = values_np[1:]
    next_values[-1] = last_val_arr

    deltas = rewards_np + gamma * (1.0 - dones_np) * next_values - values_np
    advantages = np.zeros_like(rewards_np, dtype=np.float32)
    lastgaelam = np.zeros((N,), dtype=np.float32)
    for t in reversed(range(T)):
        lastgaelam = deltas[t] + gamma * gae_lambda * (1.0 - dones_np[t]) * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values_np

    if returns.shape[1] == 1:
        return returns.reshape(-1), advantages.reshape(-1)
    return returns, advantages


def compute_rollout_targets(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    estimator: Literal["returns", "dae", "gae"],
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    if estimator == "returns":
        returns = np.asarray(
            compute_returns(rewards, dones, last_value, gamma), dtype=np.float32
        )
        values_np = np.asarray(values, dtype=np.float32)
        if values_np.ndim == 1:
            values_np = values_np.reshape(-1, 1)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        advantages = returns - values_np
        return returns, advantages
    if estimator == "dae":
        return compute_dae_targets(rewards, dones, values, last_value, gamma)
    if estimator == "gae":
        return compute_gae_targets(
            rewards, dones, values, last_value, gamma, gae_lambda
        )
    raise ValueError(f"Unknown advantage estimator: {estimator}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = int(
        (args.total_timesteps + args.batch_size - 1) // args.batch_size
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

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    obs_dim = infer_obs_dim(envs.single_observation_space)
    act_dim = int(np.prod(envs.single_action_space.shape))

    config = VMPOConfig(
        gamma=args.gamma,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        optimizer=args.optimizer,
        sgd_momentum=args.sgd_momentum,
        topk_fraction=args.topk_fraction,
        temperature_init=args.temperature_init,
        temperature_lr=args.temperature_lr,
        epsilon_eta=args.epsilon_eta,
        epsilon_mu=args.epsilon_mu,
        epsilon_sigma=args.epsilon_sigma,
        alpha_lr=args.alpha_lr,
        max_grad_norm=args.max_grad_norm,
    )

    action_low = getattr(envs.single_action_space, "low", None)
    action_high = getattr(envs.single_action_space, "high", None)
    if (
        action_low is None
        or action_high is None
        or not (np.all(np.isfinite(action_low)) and np.all(np.isfinite(action_high)))
    ):
        probe_env = gym.make(args.env_id)
        try:
            base_action_space = probe_env.action_space
            if not isinstance(base_action_space, gym.spaces.Box):
                raise ValueError("VMPO only supports continuous action spaces.")
            action_low = np.asarray(base_action_space.low, dtype=np.float32)
            action_high = np.asarray(base_action_space.high, dtype=np.float32)
        finally:
            probe_env.close()

    agent = VMPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        config=config,
        action_low=action_low,
        action_high=action_high,
        device=device,
        policy_layer_sizes=args.policy_layer_sizes,
        value_layer_sizes=args.value_layer_sizes,
    )

    obs_buf: list[np.ndarray] = []
    actions_buf: list[np.ndarray] = []
    rewards_buf: list[np.ndarray] = []
    dones_buf: list[np.ndarray] = []
    values_buf: list[np.ndarray] = []
    means_buf: list[np.ndarray] = []
    log_stds_buf: list[np.ndarray] = []

    def reset_rollout() -> None:
        obs_buf.clear()
        actions_buf.clear()
        rewards_buf.clear()
        dones_buf.clear()
        values_buf.clear()
        means_buf.clear()
        log_stds_buf.clear()

    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    global_step = 0

    try:
        while global_step < args.total_timesteps:
            action, value, mean, log_std = agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.asarray(terminated) | np.asarray(truncated)
            reward = np.asarray(reward, dtype=np.float32)

            obs_buf.append(obs)
            actions_buf.append(action)
            rewards_buf.append(reward)
            dones_buf.append(done)
            values_buf.append(value)
            means_buf.append(mean)
            log_stds_buf.append(log_std)

            obs = next_obs
            global_step += args.num_envs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = float(np.asarray(info["episode"]["r"]).reshape(-1)[0])
                        ep_len = float(np.asarray(info["episode"]["l"]).reshape(-1)[0])
                        print(f"global_step={global_step}, episodic_return={ep_return}")
                        writer.add_scalar(
                            "charts/episodic_return", ep_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", ep_len, global_step
                        )

            if len(obs_buf) >= args.num_steps:
                obs_arr = np.stack(obs_buf)
                actions_arr = np.stack(actions_buf)
                rewards_arr = np.asarray(rewards_buf, dtype=np.float32)
                dones_arr = np.asarray(dones_buf, dtype=np.float32)
                values_arr = np.asarray(values_buf, dtype=np.float32)
                means_arr = np.stack(means_buf)
                log_stds_arr = np.stack(log_stds_buf)

                last_value = agent.value(obs) * (1.0 - dones_arr[-1])

                T, N, _ = obs_arr.shape
                obs_flat = obs_arr.reshape(T * N, -1)
                actions_flat = actions_arr.reshape(T * N, -1)
                rewards_flat = rewards_arr.reshape(T, N)
                dones_flat = dones_arr.reshape(T, N)
                values_flat = values_arr.reshape(T, N)
                means_flat = means_arr.reshape(T * N, -1)
                log_stds_flat = log_stds_arr.reshape(T * N, -1)

                returns, advantages = compute_rollout_targets(
                    rewards=rewards_flat,
                    dones=dones_flat,
                    values=values_flat,
                    last_value=last_value,
                    gamma=args.gamma,
                    estimator=args.advantage_estimator,
                    gae_lambda=args.gae_lambda,
                )
                returns_flat = returns.reshape(T * N, 1)
                advantages_flat = advantages.reshape(T * N, 1)

                batch = {
                    "obs": torch.tensor(obs_flat, dtype=torch.float32, device=device),
                    "actions": torch.tensor(
                        actions_flat, dtype=torch.float32, device=device
                    ),
                    "returns": torch.tensor(
                        returns_flat, dtype=torch.float32, device=device
                    ),
                    "advantages": torch.tensor(
                        advantages_flat, dtype=torch.float32, device=device
                    ),
                    "old_means": torch.tensor(
                        means_flat, dtype=torch.float32, device=device
                    ),
                    "old_log_stds": torch.tensor(
                        log_stds_flat, dtype=torch.float32, device=device
                    ),
                }

                metrics = {}
                for _ in range(int(args.updates_per_rollout)):
                    metrics = agent.update(batch)

                for key, value in metrics.items():
                    writer.add_scalar(key, value, global_step)

                sps = int(global_step / max(1e-8, (time.time() - start_time)))
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar(
                    "charts/learning_rate_policy", args.policy_lr, global_step
                )
                writer.add_scalar(
                    "charts/learning_rate_value", args.value_lr, global_step
                )
                print("SPS:", sps)

                reset_rollout()

            if global_step >= args.total_timesteps or (args.eval_interval > 0 and global_step % args.eval_interval == 0):
                eval_metrics = evaluate(
                    agent=agent,
                    make_env=make_env,
                    env_id=args.env_id,
                    seed=args.seed + 1000,
                    gamma=args.gamma,
                    n_episodes=args.eval_episodes,
                    vectorized=True,
                )
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, global_step)
                print(
                    f"eval step={global_step} mean={eval_metrics['eval/return_mean']:.3f} "
                    f"std={eval_metrics['eval/return_std']:.3f}"
                )

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

            _, episodic_returns = evaluate(
                agent=agent,
                make_env=make_env,
                env_id=args.env_id,
                seed=args.seed + 1000,
                gamma=args.gamma,
                n_episodes=10,
                run_name=f"{run_name}-eval",
                capture_video=True,
                vectorized=True,
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
                    "VMPO",
                    f"runs/{run_name}",
                    f"videos/{run_name}-eval",
                )

    finally:
        envs.close()
        writer.close()
