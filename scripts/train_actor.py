"""Train an ActorResNet with DDPG on a target dataset at native resolution.

Adapts the Learning-to-Paint DDPG training loop for variable canvas widths.
The actor learns to paint images from the target dataset by maximising the
MSE reward (improvement in canvas-to-target distance per stroke).

Usage:
    python scripts/train_actor.py --width 32 --dataset cifar10 --data_dir /path/to/cifar10/train
    python scripts/train_actor.py --width 224 --dataset imagenet --data_dir /path/to/imagenet/train

Transfer learning: pass --pretrained_actor to initialise from an existing
checkpoint (e.g. CelebA-trained actor.pkl).  The AdaptiveAvgPool2d in the
new actor makes pretrained weights compatible across resolutions.

Reference: Huang et al., "Learning to Paint" (ICCV 2019)
"""

import argparse
import copy
import csv
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from painter.painter import ActorResNet, RendererFCN
from painter.painter_utils import decode


# ── Painting Environment ──────────────────────────────────────────────────

class PaintEnv:
    """Stroke-based painting environment at a configurable resolution.

    The environment maintains a canvas and a target image.  At each step
    the actor predicts stroke parameters, which are rendered and composited
    onto the canvas.  The reward is the normalised improvement in MSE.

    Args:
        width: Canvas side length in pixels.
        max_step: Maximum number of actor forward passes per episode.
        batch_size: Number of parallel environments.
        device: Torch device string.
    """

    def __init__(self, width: int, max_step: int, batch_size: int,
                 device: str) -> None:
        self.width = width
        self.max_step = max_step
        self.batch_size = batch_size
        self.device = device

        # Coordinate grids
        i = torch.linspace(0, 1, width, device=device).view(-1, 1).repeat(1, width)
        j = torch.linspace(0, 1, width, device=device).view(1, -1).repeat(width, 1)
        self.coord = torch.stack([i, j], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def reset(self, target_images: torch.Tensor) -> torch.Tensor:
        """Resets the canvas and sets new target images.

        Args:
            target_images: Batch of target images (B, 3, W, W) in [0, 1].

        Returns:
            Initial observation (B, 9, W, W).
        """
        self.target = target_images.to(self.device)
        self.canvas = torch.zeros_like(self.target)
        self.step_count = 0
        self.initial_dist = self._distance()
        return self._obs()

    def step(self, actions: torch.Tensor, renderer: RendererFCN
             ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Applies a stroke action and returns the next observation and reward.

        Args:
            actions: Stroke parameters (B, 65).
            renderer: RendererFCN for rendering strokes.

        Returns:
            Tuple of (next_obs, reward, done).
        """
        prev_dist = self._distance()
        self.canvas, _ = decode(actions, self.canvas, renderer, self.width)
        self.step_count += 1
        curr_dist = self._distance()

        # Reward = normalised improvement
        reward = (prev_dist - curr_dist) / (self.initial_dist + 1e-8)
        done = self.step_count >= self.max_step
        return self._obs(), reward, done

    def _distance(self) -> torch.Tensor:
        """Per-sample MSE between canvas and target."""
        return ((self.canvas - self.target) ** 2).mean(dim=(1, 2, 3))

    def _obs(self) -> torch.Tensor:
        """Builds the 9-channel observation."""
        B = self.target.shape[0]
        T = torch.ones(B, 1, self.width, self.width, device=self.device)
        stepnum = T * (self.step_count / self.max_step)
        return torch.cat([self.canvas, self.target, stepnum, self.coord], dim=1)


# ── Replay Buffer ─────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.stack(s), torch.stack(a), torch.stack(r),
                torch.stack(ns), torch.stack(d))

    def __len__(self):
        return len(self.buffer)


# ── Critic Network ────────────────────────────────────────────────────────

class Critic(torch.nn.Module):
    """Simple CNN critic that estimates Q(state, action)."""

    def __init__(self, width: int):
        super().__init__()
        # Input: obs (9 channels) + next_canvas (3 channels) = 12
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(12, 32, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, obs, next_canvas):
        x = torch.cat([obs, next_canvas], dim=1)
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)


# ── Dataset Loading ───────────────────────────────────────────────────────

def load_images(data_dir: str, width: int, max_images: int = 50000) -> torch.Tensor:
    """Loads images from a directory and resizes to width×width.

    Args:
        data_dir: Root directory with class subdirectories of images.
        width: Target resolution.
        max_images: Maximum number of images to load.

    Returns:
        Float tensor of shape (N, 3, width, width) in [0, 1].
    """
    transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.ToTensor(),
    ])
    images = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                img = Image.open(path).convert('RGB')
                images.append(transform(img))
                if len(images) >= max_images:
                    return torch.stack(images)
    return torch.stack(images) if images else torch.empty(0)


# ── Validation ────────────────────────────────────────────────────────────

@torch.inference_mode()
def _evaluate_painting(actor: ActorResNet, renderer: RendererFCN,
                       env: PaintEnv, val_images: torch.Tensor,
                       width: int, device: str, max_step: int,
                       max_val_batches: int = 5) -> float:
    """Paints validation images and returns mean MSE to targets.

    Args:
        actor: Actor in eval mode.
        renderer: Trained renderer.
        env: Painting environment (reused for coord grids).
        val_images: Held-out images (N, 3, W, W) in [0, 1].
        width: Canvas resolution.
        device: Torch device.
        max_step: Number of painting steps.
        max_val_batches: Max batches to evaluate (caps compute).

    Returns:
        Mean MSE distance between final canvases and targets.
    """
    actor.eval()
    batch_size = min(len(val_images), env.batch_size)
    total_dist = 0.0
    n_batches = 0

    for start in range(0, len(val_images), batch_size):
        if n_batches >= max_val_batches:
            break
        batch = val_images[start:start + batch_size].to(device)
        if len(batch) < 2:
            continue  # skip tiny batches (BatchNorm needs >1)

        # Paint the batch
        canvas = torch.zeros_like(batch)
        T = torch.ones(len(batch), 1, width, width, device=device)
        i_g = torch.linspace(0, 1, width, device=device).view(-1, 1).repeat(1, width)
        j_g = torch.linspace(0, 1, width, device=device).view(1, -1).repeat(width, 1)
        coord = torch.stack([i_g, j_g], dim=0).unsqueeze(0).expand(len(batch), -1, -1, -1)

        for step in range(max_step):
            stepnum = T * (step / max_step)
            obs = torch.cat([canvas, batch, stepnum, coord], dim=1)
            actions = actor(obs)
            canvas, _ = decode(actions, canvas, renderer, width)

        mse = ((canvas - batch) ** 2).mean().item()
        total_dist += mse
        n_batches += 1

    actor.train()
    return total_dist / max(n_batches, 1)


# ── DDPG Training Loop ────────────────────────────────────────────────────

def train(args):
    device = args.device
    width = args.width

    print(f'Training ActorResNet(width={width}) with DDPG')
    print(f'  Dataset: {args.data_dir}')
    print(f'  Device: {device}')

    # Load dataset with train/val split
    print('Loading images...')
    all_images = load_images(args.data_dir, width, max_images=args.max_images)
    n_total = len(all_images)
    n_val = max(1, int(n_total * args.val_fraction))
    perm = torch.randperm(n_total)
    val_images = all_images[perm[:n_val]]
    train_images = all_images[perm[n_val:]]
    print(f'  Loaded {n_total} images → {len(train_images)} train, {n_val} val')

    # Models
    actor = ActorResNet(width=width).to(device)
    actor_target = copy.deepcopy(actor)
    renderer = RendererFCN(width=width).to(device)

    # Load pretrained weights if provided
    if args.pretrained_actor:
        print(f'  Loading pretrained actor from {args.pretrained_actor}')
        actor.load_state_dict(torch.load(args.pretrained_actor, map_location=device))
        actor_target.load_state_dict(actor.state_dict())

    if args.pretrained_renderer:
        print(f'  Loading pretrained renderer from {args.pretrained_renderer}')
        renderer.load_state_dict(torch.load(args.pretrained_renderer, map_location=device))
    else:
        raise ValueError('--pretrained_renderer is required (train renderer first)')

    renderer.eval()

    critic = Critic(width).to(device)
    critic_target = copy.deepcopy(critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    env = PaintEnv(width, args.env_max_step, args.env_batch, device)
    replay = ReplayBuffer(args.replay_size)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'actor_{width}.pkl')

    # Metrics CSV for paper plots
    results_dir = os.path.dirname(save_path)
    metrics_csv_path = os.path.join(results_dir, f'training_metrics_{width}.csv')
    metrics_file = open(metrics_csv_path, 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow([
        'episode', 'elapsed_sec', 'episode_reward', 'mean_critic_loss',
        'mean_actor_loss', 'mean_q_value', 'noise_scale',
        'train_canvas_mse', 'val_canvas_mse', 'best_val_mse',
    ])
    print(f'  Metrics CSV: {metrics_csv_path}', flush=True)

    tau = args.tau
    gamma = args.gamma
    noise_scale = args.noise_scale

    best_val_dist = float('inf')

    # Running accumulators for averaging over log_interval episodes
    acc_reward = 0.0
    acc_critic_loss = 0.0
    acc_actor_loss = 0.0
    acc_q_value = 0.0
    acc_train_mse = 0.0
    acc_updates = 0

    t0 = time.time()

    for episode in range(1, args.max_episodes + 1):
        # Sample batch of target images from TRAIN set
        idx = torch.randint(0, len(train_images), (args.env_batch,))
        targets = train_images[idx].to(device)
        obs = env.reset(targets)

        episode_reward = 0.0
        done = False
        ep_critic_loss = 0.0
        ep_actor_loss = 0.0
        ep_q_value = 0.0
        ep_updates = 0

        while not done:
            # Actor predicts actions + exploration noise
            with torch.no_grad():
                actions = actor(obs)
                noise = torch.randn_like(actions) * noise_scale
                actions = (actions + noise).clamp(0, 1)

            next_obs, reward, done = env.step(actions, renderer)
            episode_reward += reward.mean().item()

            # Store transitions
            for b in range(args.env_batch):
                replay.push(obs[b].cpu(), actions[b].cpu(), reward[b:b+1].cpu(),
                            next_obs[b].cpu(), torch.tensor([float(done)]))

            obs = next_obs

            # Train if enough samples
            if len(replay) >= args.batch_size:
                for _ in range(args.train_times):
                    s, a, r, ns, d = replay.sample(args.batch_size)
                    s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)

                    # Critic update
                    with torch.no_grad():
                        next_a = actor_target(ns)
                        next_canvas, _ = decode(next_a, ns[:, :3], renderer, width)
                        target_q = r + gamma * (1 - d) * critic_target(ns, next_canvas)

                    current_canvas, _ = decode(a, s[:, :3], renderer, width)
                    current_q = critic(s, current_canvas)
                    c_loss = F.mse_loss(current_q, target_q)

                    critic_opt.zero_grad()
                    c_loss.backward()
                    critic_opt.step()

                    # Actor update
                    pred_a = actor(s)
                    pred_canvas, _ = decode(pred_a, s[:, :3], renderer, width)
                    a_loss = -critic(s, pred_canvas).mean()

                    actor_opt.zero_grad()
                    a_loss.backward()
                    actor_opt.step()

                    # Soft update targets
                    for p, tp in zip(actor.parameters(), actor_target.parameters()):
                        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
                    for p, tp in zip(critic.parameters(), critic_target.parameters()):
                        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

                    # Accumulate step-level metrics
                    ep_critic_loss += c_loss.item()
                    ep_actor_loss += a_loss.item()
                    ep_q_value += current_q.mean().item()
                    ep_updates += 1

        # Compute train canvas MSE at end of episode
        with torch.no_grad():
            train_mse = ((env.canvas - env.target) ** 2).mean().item()

        # Accumulate episode-level metrics
        acc_reward += episode_reward
        acc_train_mse += train_mse
        if ep_updates > 0:
            acc_critic_loss += ep_critic_loss / ep_updates
            acc_actor_loss += ep_actor_loss / ep_updates
            acc_q_value += ep_q_value / ep_updates
            acc_updates += 1

        # Log metrics every log_interval episodes
        if episode % args.log_interval == 0:
            elapsed = time.time() - t0
            n = args.log_interval
            avg_reward = acc_reward / n
            avg_train_mse = acc_train_mse / n
            avg_critic = acc_critic_loss / max(acc_updates, 1)
            avg_actor = acc_actor_loss / max(acc_updates, 1)
            avg_q = acc_q_value / max(acc_updates, 1)

            print(f'  Episode {episode}/{args.max_episodes}  '
                  f'reward={avg_reward:.4f}  critic_loss={avg_critic:.4f}  '
                  f'actor_loss={avg_actor:.4f}  Q={avg_q:.4f}  '
                  f'train_mse={avg_train_mse:.6f}  noise={noise_scale:.4f}  '
                  f'{elapsed:.0f}s', flush=True)

            # Write to CSV (val_mse written as empty unless val episode)
            metrics_writer.writerow([
                episode, f'{elapsed:.1f}', f'{avg_reward:.6f}',
                f'{avg_critic:.6f}', f'{avg_actor:.6f}', f'{avg_q:.6f}',
                f'{noise_scale:.6f}', f'{avg_train_mse:.6f}', '', '',
            ])
            metrics_file.flush()

            # Reset accumulators
            acc_reward = 0.0
            acc_critic_loss = 0.0
            acc_actor_loss = 0.0
            acc_q_value = 0.0
            acc_train_mse = 0.0
            acc_updates = 0

        # Validation: measure painting quality on held-out images
        if episode % args.val_interval == 0:
            val_dist = _evaluate_painting(actor, renderer, env, val_images,
                                          width, device, args.env_max_step)
            improved = val_dist < best_val_dist
            if improved:
                best_val_dist = val_dist
                best_path = save_path.replace('.pkl', '_best.pkl')
                torch.save(actor.state_dict(), best_path)
            print(f'  [VAL] Episode {episode}  val_mse={val_dist:.6f}  '
                  f'best={best_val_dist:.6f}  {"★ new best" if improved else ""}',
                  flush=True)

            # Write val row to CSV
            metrics_writer.writerow([
                episode, f'{time.time() - t0:.1f}', '', '', '', '', '',
                '', f'{val_dist:.6f}', f'{best_val_dist:.6f}',
            ])
            metrics_file.flush()

        if episode % args.save_interval == 0:
            torch.save(actor.state_dict(), save_path)
            print(f'  Saved to {save_path}', flush=True)

        # Decay noise
        noise_scale *= args.noise_decay

    metrics_file.close()
    torch.save(actor.state_dict(), save_path)
    print(f'Training complete. Final model: {save_path}')
    print(f'Metrics saved to: {metrics_csv_path}')


def main():
    parser = argparse.ArgumentParser(description='Train ActorResNet with DDPG')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--pretrained_actor', type=str, default=None)
    parser.add_argument('--pretrained_renderer', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='models/painter_actor')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_episodes', type=int, default=20000)
    parser.add_argument('--env_max_step', type=int, default=40)
    parser.add_argument('--env_batch', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--replay_size', type=int, default=32000)
    parser.add_argument('--train_times', type=int, default=10)
    parser.add_argument('--actor_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95 ** 5)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--noise_scale', type=float, default=0.3)
    parser.add_argument('--noise_decay', type=float, default=0.9999)
    parser.add_argument('--max_images', type=int, default=50000)
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of images held out for validation')
    parser.add_argument('--val_interval', type=int, default=200,
                        help='Run validation every N episodes')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=500)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
