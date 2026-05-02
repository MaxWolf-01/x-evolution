# /// script
# dependencies = [
#     "fire",
#     "gymnasium[box2d]>=1.0.0",
#     "gymnasium[other]",
#     "x-evolution>=0.0.20",
#     "x-transformers",
#     "wandb"
# ]
# ///

import re
import fire
from shutil import rmtree
from itertools import cycle
from collections import deque

import numpy as np
import gymnasium as gym
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from x_evolution import EvoStrategy
from x_transformers import Decoder

# helpers

def exists(v):
    return v is not None

# schedule parsing

def parse_string_schedule(schedule_str):
    schedule_str = re.sub(r'\(([^)]+)\)\s*\*\s*(\d+)', lambda m: f" {m.group(1)} " * int(m.group(2)), schedule_str)

    phases = []
    for duration, phase in re.findall(r'(\d+)\s*(all|both|inner|outer)', schedule_str.lower()):
        phase = 'all' if phase == 'both' else phase
        phases.extend([phase] * int(duration))

    assert len(phases) > 0, 'could not parse phase schedule string'
    return phases

# orthogonal update from inner to outer

def orthogonal_project(x, residual):
    dtype = residual.dtype
    residual, x = residual.double(), x.double()

    unit = F.normalize(residual, dim = -1)
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return orthogonal.to(dtype)

# hierarchical transformer
# outer pre -> inner (residual gated every inner_update_every steps) -> outer post

class HierarchicalTransformer(Module):
    def __init__(
        self,
        dim_in,
        dim,
        num_actions,
        outer_depth = 1,
        inner_depth = 1,
        inner_update_every = 1
    ):
        super().__init__()
        self.inner_update_every = inner_update_every

        self.inner_update_emb = nn.Parameter(torch.zeros(dim))

        self.token_emb = nn.Linear(dim_in, dim)

        decoder_kwargs = dict(
            dim = dim,
            attn_dim_head = 32,
            rotary_pos_emb = True,
            rotary_emb_dim = 32,
            pre_norm_has_final_norm = False
        )

        self.outer_pre = Decoder(depth = outer_depth, **decoder_kwargs)

        self.inner = Decoder(depth = inner_depth, **decoder_kwargs)
        self.inner_gru_norm = nn.RMSNorm(dim)
        self.inner_gru = nn.GRU(dim, dim, batch_first = True)

        self.outer_post = Decoder(depth = outer_depth, **decoder_kwargs)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_actions)
        )

    def forward(self, x, step = 0, cache = None):
        b, n, _ = x.shape
        assert n == 1, 'only single token rollouts are supported'

        x = self.token_emb(x)

        cache_pre, cache_inner, cache_post, cache_gru, last_inner_update = cache if exists(cache) else (None, None, None, None, None)

        # outer pre

        x, cache_pre = self.outer_pre(x, cache = cache_pre, return_hiddens = True)

        should_update = (step % self.inner_update_every) == 0

        # let the inner network know when it is an update step

        x_inner = x

        if should_update:
            x_inner = x_inner + self.inner_update_emb

        # inner - always runs for KV context, but residual only applied every inner_update_every steps

        x_inner, cache_inner = self.inner(x_inner, cache = cache_inner, return_hiddens = True)

        x_inner_gru, cache_gru = self.inner_gru(self.inner_gru_norm(x_inner), cache_gru)
        x_inner = x_inner + x_inner_gru

        if should_update:
            last_inner_update = orthogonal_project(x_inner, residual = x)

        if exists(last_inner_update):
            x = x + last_inner_update

        # outer post

        x, cache_post = self.outer_post(x, cache = cache_post, return_hiddens = True)

        return self.to_logits(x), (cache_pre, cache_inner, cache_post, cache_gru, last_inner_update)

# environment

class LunarEnvironment(Module):
    def __init__(
        self,
        video_folder = './recordings',
        render_every_eps = 500,
        max_steps = 500,
        repeats = 1,
        vectorized = False,
        num_envs = 1,
        rolling_window = 20
    ):
        super().__init__()
        self.vectorized = vectorized
        self.num_envs = num_envs

        if vectorized:
            env = gym.make_vec('LunarLander-v3', num_envs = num_envs, render_mode = 'rgb_array')
        else:
            env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

        self.env = env
        self.max_steps = max_steps
        self.repeats = repeats
        self.video_folder = video_folder
        self.render_every_eps = render_every_eps
        self._pre_main_callback_called = False

        self.last_steps = deque(maxlen = rolling_window)

    def pre_main_callback(self):
        if self._pre_main_callback_called:
            return

        self._pre_main_callback_called = True
        rmtree(self.video_folder, ignore_errors = True)

        if not self.vectorized:
            self.env = gym.wrappers.RecordVideo(
                env = self.env,
                video_folder = self.video_folder,
                name_prefix = 'recording',
                episode_trigger = lambda eps_num: (eps_num % self.render_every_eps) == 0,
                disable_logger = True
            )

    @property
    def avg_steps(self):
        if len(self.last_steps) == 0:
            return 0.
        return sum(self.last_steps) / len(self.last_steps)

    def forward(self, model):
        device = next(model.parameters()).device
        seed = torch.randint(0, int(1e6), ())

        num_envs = self.num_envs if self.vectorized else 1
        cum_reward = torch.zeros(num_envs, device = device)

        for _ in range(self.repeats):
            state, _ = self.env.reset(seed = seed.item())

            step = 0
            dones = torch.zeros(num_envs, device = device, dtype = torch.bool)
            cache = None

            while step < self.max_steps and not dones.all():
                state_torch = torch.from_numpy(state).to(device)

                if not self.vectorized:
                    state_torch = state_torch.unsqueeze(0)

                state_torch = state_torch.unsqueeze(1)

                action_logits, cache = model(state_torch, step = step, cache = cache)
                action_logits = action_logits[:, -1, :]

                action = F.gumbel_softmax(action_logits, hard = True).argmax(dim = -1)

                env_action = action.detach().cpu().numpy() if self.vectorized else action.item()
                next_state, reward, truncated, terminated, *_ = self.env.step(env_action)

                reward_np = np.array(reward) if not isinstance(reward, np.ndarray) else reward
                total_reward = torch.from_numpy(reward_np).float().to(device)

                mask = (~dones).float()
                cum_reward += total_reward * mask

                dones_np = np.array(truncated | terminated) if not isinstance(truncated | terminated, np.ndarray) else (truncated | terminated)
                dones |= torch.from_numpy(dones_np).to(device)

                step += 1
                state = next_state

            self.last_steps.append(step)

        if not self.vectorized:
            return cum_reward.item() / self.repeats

        return cum_reward / self.repeats

# main

def main(
    vectorized = False,
    num_envs = 8,
    cpu = False,
    phase_schedule = '150all (50inner 50all)*10',
    inner_update_every = 1,
    outer_depth = 1,
    inner_depth = 1,
    dim = 32,
    use_wandb = True,
    wandb_project = 'lunar-hierarchical-transformer',
    rolling_window = 20,
    noise_population_size = 50,
    learning_rate = 1e-3,
    noise_scale = 1e-2
):
    if use_wandb:
        wandb.init(project = wandb_project, config = locals())

    model = HierarchicalTransformer(
        dim_in = 8,
        dim = dim,
        num_actions = 4,
        outer_depth = outer_depth,
        inner_depth = inner_depth,
        inner_update_every = inner_update_every
    )

    env = LunarEnvironment(
        repeats = 2,
        vectorized = vectorized,
        num_envs = num_envs,
        rolling_window = rolling_window
    )

    # partition parameters into inner vs outer

    inner_param_ids = {id(p) for p in model.inner.parameters()} | {id(model.inner_update_emb)} | {id(p) for p in model.inner_gru.parameters()} | {id(p) for p in model.inner_gru_norm.parameters()}

    inner_params = [p for p in model.parameters() if id(p) in inner_param_ids]
    outer_params = [p for p in model.parameters() if id(p) not in inner_param_ids]
    all_params = list(model.parameters())


    evo_kwargs = dict(
        environment = env,
        vectorized = vectorized,
        vector_size = num_envs,
        cpu = cpu,
        num_generations = 1,
        noise_population_size = noise_population_size,
        noise_low_rank = 2,
        noise_scale = noise_scale,
        noise_scale_clamp_range = (5e-3, 2e-2),
        learned_noise_scale = True,
        use_sigma_optimizer = True,
        learning_rate = learning_rate,
        noise_scale_learning_rate = 1e-4,
        use_scheduler = False,
        verbose = False,
        sync_on_init = True
    )

    print('Setting up EvoStrategy wrappers...')

    evos = dict(
        all = EvoStrategy(model, params_to_optimize = all_params, **evo_kwargs),
        inner = EvoStrategy(model, params_to_optimize = inner_params, **evo_kwargs),
        outer = EvoStrategy(model, params_to_optimize = outer_params, **evo_kwargs),
    )

    # schedule

    phases = parse_string_schedule(phase_schedule)
    total_generations = len(phases)
    phase_gen = cycle(phases)

    print('\n--- Training Phase Schedule ---')
    print(f'Schedule: {phase_schedule}')
    print(f'Total Generations: {total_generations}')
    print(f'Inner updates every: {inner_update_every} steps')
    print('-------------------------------\n')

    pbar = tqdm(total = total_generations, desc = 'Generations')
    running_rewards = deque(maxlen = rolling_window)

    for gen in range(1, total_generations + 1):
        phase = next(phase_gen)
        evo = evos[phase]

        fitnesses = evo(num_generations = 1, verbose = False)
        avg_fit = fitnesses.mean().item()

        running_rewards.append(avg_fit)
        avg_reward = sum(running_rewards) / len(running_rewards)
        avg_steps = env.avg_steps

        pbar.set_postfix(phase = phase, avg_reward = round(avg_reward, 2), avg_steps = round(avg_steps, 1))
        pbar.update(1)

        if use_wandb:
            wandb.log(dict(
                generation = gen,
                phase = phase,
                avg_fitness = avg_fit,
                avg_reward_window = avg_reward,
                avg_steps = avg_steps,
            ))

    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    fire.Fire(main)
