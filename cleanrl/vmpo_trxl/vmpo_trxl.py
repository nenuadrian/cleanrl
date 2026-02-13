import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import memory_gym  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from einops import rearrange
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from pom_env import PoMEnv  # noqa
from torch.distributions import Categorical, kl_divergence
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

    # Algorithm specific arguments
    env_id: str = "MortarMayhem-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 200000000
    """total timesteps of the experiments"""
    init_lr: float = 2.75e-4
    """the initial learning rate of the optimizer"""
    final_lr: float = 1.0e-5
    """the final learning rate of the optimizer after linearly annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_steps: int = 32 * 512 * 10000
    """the number of steps to linearly anneal the learning rate from initial to final"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    vmpo_topk_fraction: float = 0.5
    """fraction of highest advantages used in the E-step weights"""
    vmpo_eps_eta: float = 0.02
    """temperature dual constraint from V-MPO (epsilon_eta)"""
    vmpo_eps_alpha: float = 0.01
    """policy KL dual constraint from V-MPO (epsilon_alpha)"""
    vmpo_init_eta: float = 1.0
    """initial value of the temperature dual variable eta"""
    vmpo_init_alpha: float = 1.0
    """initial value of the KL dual variable alpha"""
    vmpo_min_eta: float = 1e-8
    """minimum value of eta to keep it positive"""
    vmpo_min_alpha: float = 1e-8
    """minimum value of alpha to keep it positive"""
    vmpo_dual_lr: float = 1e-4
    """learning rate for the dual variables eta and alpha"""
    vmpo_dual_steps: int = 1
    """number of dedicated dual-variable gradient steps per minibatch"""
    init_ent_coef: float = 0.0001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after linearly annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 3
    """the number of transformer layers"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 119
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = "absolute"
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """
    reconstruction_coef: float = 0.0
    """the coefficient of the observation reconstruction loss, if set to 0.0 the reconstruction loss is not used"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, render_mode="debug_rgb_array"):
    if "MiniGrid" in env_id:
        if render_mode == "debug_rgb_array":
            render_mode = "rgb_array"

    def thunk():
        if "MiniGrid" in env_id:
            env = gym.make(env_id, agent_view_size=3, tile_size=28, render_mode=render_mode)
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
            env = gym.wrappers.TimeLimit(env, 96)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert self.head_size * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))  # -inf causes NaN

        # Normalize energy values and apply softmax to retrieve the attention scores
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_size)

        return self.fc_out(out), attention


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # Pre-layer normalization (post-layer normalization is usually less effective)
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # K = V -> self-attention
        attention, attention_weights = self.attention(value, key, query_, mask)  # MHA
        x = attention + query  # Skip connection
        x_ = self.layer_norm_attn(x)  # Pre-layer normalization
        forward = self.fc_projection(x_)  # Forward projection
        out = forward + x  # Skip connection
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        self.transformer_layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices):
        # Add positional encoding to every transformer layer input
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        # Forward transformer layers and return new memories (i.e. hidden states)
        out_memories = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, attention_weights = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )  # args: value, key, query, mask
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, max_episode_steps):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.obs_shape) > 1:
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
        else:
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.trxl_dim))

        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding
        )

        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
                for num_actions in action_space_shape
            ]
        )
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )

    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None, return_logits=False):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        self.x = x
        logits = [branch(x) for branch in self.actor_branches]
        probs = [Categorical(logits=branch_logits) for branch_logits in logits]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
        log_probs = []
        for i, dist in enumerate(probs):
            log_probs.append(dist.log_prob(action[:, i]))
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
        outputs = (action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), memory)
        if return_logits:
            return outputs + (logits,)
        return outputs

    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x.permute((0, 2, 3, 1))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.vmpo_min_eta = max(args.vmpo_min_eta, 1e-12)
    args.vmpo_min_alpha = max(args.vmpo_min_alpha, 1e-12)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine the device to be used for training and set the default tensor type
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    observation_space = envs.single_observation_space
    action_space_shape = (
        (envs.single_action_space.n,)
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else tuple(envs.single_action_space.nvec)
    )
    env_ids = range(args.num_envs)
    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
    # Determine maximum episode steps
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    if not max_episode_steps:
        envs.envs[0].reset()  # Memory Gym envs need to be reset before accessing max_episode_steps
        max_episode_steps = envs.envs[0].max_episode_steps
    if max_episode_steps <= 0:
        max_episode_steps = 1024  # Memory Gym envs have max_episode_steps set to -1
    # Set transformer memory length to max episode steps if greater than max episode steps
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

    agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
    log_eta = nn.Parameter(torch.log(torch.tensor(args.vmpo_init_eta, dtype=torch.float32)))
    log_alpha = nn.Parameter(torch.log(torch.tensor(args.vmpo_init_alpha, dtype=torch.float32)))
    policy_optimizer = optim.AdamW(agent.parameters(), lr=args.init_lr)
    dual_optimizer = optim.AdamW([log_eta, log_alpha], lr=args.vmpo_dual_lr)
    bce_loss = nn.BCELoss()  # Binary cross entropy loss for observation reconstruction

    # ALGO Logic: Storage setup
    rewards = torch.zeros((args.num_steps, args.num_envs))
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
    dones = torch.zeros((args.num_steps, args.num_envs))
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
    values = torch.zeros((args.num_steps, args.num_envs))
    old_policy_logits = [
        torch.zeros((args.num_steps, args.num_envs, num_actions), dtype=torch.float32) for num_actions in action_space_shape
    ]
    # The length of stored-memories is equal to the number of sampled episodes during training data sampling
    # (num_episodes, max_episode_length, num_layers, embed_dim)
    stored_memories = []
    # Memory mask used during attention
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    # Index to select the correct episode memory from stored_memories
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    # Indices to slice the episode memories into windows
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    episode_infos = deque(maxlen=100)  # Store episode results for monitoring statistics
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs)
    # Setup placeholders for each environments's current episodic memory
    next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
    # Generate episodic memory mask used in attention
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
    """ e.g. memory mask tensor looks like this if memory_length = 6
    0, 0, 0, 0, 0, 0
    1, 0, 0, 0, 0, 0
    1, 1, 0, 0, 0, 0
    1, 1, 1, 0, 0, 0
    1, 1, 1, 1, 0, 0
    1, 1, 1, 1, 1, 0
    """
    # Setup memory window indices to support a sliding window over the episodic memory
    repetitions = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
    ).long()
    memory_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    1, 2, 3, 4
    2, 3, 4, 5
    3, 4, 5, 6
    """

    for iteration in range(1, args.num_iterations + 1):
        sampled_episode_infos = []

        # Annealing the learning rate and entropy coefficient if instructed to do so
        do_anneal = args.anneal_steps > 0 and global_step < args.anneal_steps
        frac = 1 - global_step / args.anneal_steps if do_anneal else 0
        lr = (args.init_lr - args.final_lr) * frac + args.final_lr
        for param_group in policy_optimizer.param_groups:
            param_group["lr"] = lr
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

        # Init episodic memory buffer using each environments' current episodic memory
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        for step in range(args.num_steps):
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                obs[step] = next_obs
                dones[step] = next_done
                stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
                stored_memory_indices[step] = memory_indices[env_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, _, _, value, new_memory, logits = agent.get_action_and_value(
                    next_obs,
                    memory_window,
                    stored_memory_masks[step],
                    stored_memory_indices[step],
                    return_logits=True,
                )
                next_memory[env_ids, env_current_episode_step] = new_memory
                # Store actions, values, and target policy logits in the buffer
                actions[step], values[step] = action, value
                for branch_idx, branch_logits in enumerate(logits):
                    old_policy_logits[branch_idx][step] = branch_logits

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Reset and process episodic memory if done
            for id, done in enumerate(next_done):
                if done:
                    # Reset the environment's current timestep
                    env_current_episode_step[id] = 0
                    # Break the reference to the environment's episodic memory
                    mem_index = stored_memory_index[step, id]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    # Reset episodic memory
                    next_memory[id] = torch.zeros(
                        (max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32
                    )
                    if step < args.num_steps - 1:
                        # Store memory inside the buffer
                        stored_memories.append(next_memory[id])
                        # Store the reference of to the current episodic memory inside the buffer
                        stored_memory_index[step + 1 :, id] = len(stored_memories) - 1
                else:
                    # Increment environment timestep if not done
                    env_current_episode_step[id] += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        sampled_episode_infos.append(info["episode"])

        # Bootstrap value if not done
        with torch.no_grad():
            start = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start[b], end[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices)  # Retrieve the memory window from the entire episode
            next_value = agent.get_value(
                next_obs,
                memory_window,
                memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                stored_memory_indices[-1],
            )
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_old_policy_logits = [branch_logits.reshape(-1, branch_logits.shape[-1]) for branch_logits in old_policy_logits]
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
        b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
        stored_memories = torch.stack(stored_memories, dim=0)

        # Remove unnecessary padding from TrXL memory, if applicable
        actual_max_episode_steps = (stored_memory_indices * stored_memory_masks).max().item() + 1
        if actual_max_episode_steps < args.trxl_memory_length:
            b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
            b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
            stored_memories = stored_memories[:, :actual_max_episode_steps]

        # Optimizing the policy and value network
        pg_loss = torch.tensor(0.0)
        eta_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        v_loss = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)
        r_loss = torch.tensor(0.0)
        loss = torch.tensor(0.0)
        policy_kl = torch.tensor(0.0)
        temperature = torch.tensor(0.0)
        alpha_value = torch.tensor(0.0)
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_memories = stored_memories[b_memory_index[mb_inds]]
                mb_memory_windows = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])

                _, newlogprob, entropy, newvalue, _, new_logits = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mb_memory_windows,
                    b_memory_mask[mb_inds],
                    b_memory_indices[mb_inds],
                    b_actions[mb_inds],
                    return_logits=True,
                )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # V-MPO E-step: keep only top-k advantages and build normalized weights.
                topk_count = max(1, int(args.vmpo_topk_fraction * mb_advantages.shape[0]))
                topk_count = min(topk_count, mb_advantages.shape[0])
                topk_indices = torch.topk(mb_advantages, k=topk_count, sorted=False).indices
                top_advantages = mb_advantages[topk_indices].detach()

                # Joint log-prob for factorized multi-discrete action spaces.
                joint_log_prob = newlogprob.sum(dim=1)

                eta = torch.exp(log_eta).clamp_min(args.vmpo_min_eta)
                alpha = torch.exp(log_alpha).clamp_min(args.vmpo_min_alpha)

                weights = torch.softmax(top_advantages / eta.detach(), dim=0).detach()
                pg_loss = -(weights * joint_log_prob[topk_indices]).sum()

                eta_loss = eta * args.vmpo_eps_eta + eta * (
                    torch.logsumexp(top_advantages / eta, dim=0) - np.log(topk_count)
                )

                old_dists = [Categorical(logits=branch_logits[mb_inds]) for branch_logits in b_old_policy_logits]
                new_dists = [Categorical(logits=branch_logits) for branch_logits in new_logits]
                policy_kl = torch.stack(
                    [kl_divergence(old_dist, new_dist) for old_dist, new_dist in zip(old_dists, new_dists)],
                    dim=1,
                ).sum(dim=1)
                policy_kl = policy_kl.mean()

                alpha_constraint_loss = alpha * (args.vmpo_eps_alpha - policy_kl.detach())
                alpha_loss = alpha_constraint_loss + alpha.detach() * policy_kl

                v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                # Combined policy/value loss with dual variables treated as constants.
                policy_loss = pg_loss + alpha.detach() * policy_kl - ent_coef * entropy_loss + args.vf_coef * v_loss

                # Add reconstruction loss if used
                r_loss = torch.tensor(0.0, device=device)
                if args.reconstruction_coef > 0.0:
                    r_loss = bce_loss(agent.reconstruct_observation(), b_obs[mb_inds] / 255.0)
                    policy_loss += args.reconstruction_coef * r_loss
                loss = policy_loss + eta_loss + alpha_constraint_loss

                policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
                policy_optimizer.step()

                # Coordinate-style update for dual variables (eta, alpha).
                for _ in range(args.vmpo_dual_steps):
                    dual_optimizer.zero_grad()
                    eta_dual = torch.exp(log_eta).clamp_min(args.vmpo_min_eta)
                    alpha_dual = torch.exp(log_alpha).clamp_min(args.vmpo_min_alpha)
                    dual_loss = eta_dual * args.vmpo_eps_eta + eta_dual * (
                        torch.logsumexp(top_advantages / eta_dual, dim=0) - np.log(topk_count)
                    ) + alpha_dual * (args.vmpo_eps_alpha - policy_kl.detach())
                    dual_loss.backward()
                    dual_optimizer.step()
                    with torch.no_grad():
                        log_eta.data.clamp_(min=np.log(args.vmpo_min_eta))
                        log_alpha.data.clamp_(min=np.log(args.vmpo_min_alpha))

                with torch.no_grad():
                    temperature = torch.exp(log_eta)
                    alpha_value = torch.exp(log_alpha)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log and monitor training statistics
        episode_infos.extend(sampled_episode_infos)
        episode_result = {}
        if len(episode_infos) > 0:
            for key in episode_infos[0].keys():
                episode_result[key + "_mean"] = np.mean([info[key] for info in episode_infos])

        print(
            "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} eta_loss={:.3f} alpha_loss={:.3f} v_loss={:.3f} kl={:.4f} eta={:.4f} alpha={:.4f} r_loss={:.3f} value={:.3f} adv={:.3f}".format(
                iteration,
                int(global_step / (time.time() - start_time)),
                episode_result["r_mean"],
                episode_result["l_mean"],
                pg_loss.item(),
                eta_loss.item(),
                alpha_loss.item(),
                v_loss.item(),
                policy_kl.item(),
                temperature.item(),
                alpha_value.item(),
                r_loss.item(),
                torch.mean(values),
                torch.mean(advantages),
            )
        )

        if episode_result:
            for key in episode_result:
                writer.add_scalar("episode/" + key, episode_result[key], global_step)
        writer.add_scalar("episode/value_mean", torch.mean(values), global_step)
        writer.add_scalar("episode/advantage_mean", torch.mean(advantages), global_step)
        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/eta_loss", eta_loss.item(), global_step)
        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/reconstruction_loss", r_loss.item(), global_step)
        writer.add_scalar("losses/policy_kl", policy_kl.item(), global_step)
        writer.add_scalar("losses/eta", temperature.item(), global_step)
        writer.add_scalar("losses/alpha", alpha_value.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": agent.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    writer.close()
    envs.close()
