"""Deep Q-Network baseline (Double-DQN targets, experience replay, target network).

Usage:  python -m mining_rl.dqn --seed 1 --total-steps 5000000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mining_rl.common import Logger, env_kwargs, save_checkpoint, set_seed, to_torch
from mining_rl.env import VecEnv, make_env_fns
from mining_rl.networks import QNetwork


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN mining agent")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--replay-ratio", type=float, default=0.125,
                   help="gradient updates per environment step")
    p.add_argument("--target-update", type=int, default=10_000, help="env steps between target syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-fraction", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    # environment / ablations
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--no-tools", dest="tools", action="store_false")
    p.add_argument("--survival", action="store_true")
    p.add_argument("--no-inventory", dest="use_inventory", action="store_false")
    p.add_argument("--no-damage-penalty", action="store_true")
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ReplayBuffer:
    """Lives on the training device. Stores the full frame stack for s, but only the
    newest RGB frame for s' (s' is s shifted by one frame), which saves ~40% memory."""

    def __init__(self, capacity, obs_shape, inv_dim, device):
        c, h, w = obs_shape
        self.capacity, self.size, self.pos, self.device = capacity, 0, 0, device
        zeros = lambda *shape, dtype=torch.float32: torch.zeros(shape, dtype=dtype, device=device)
        self.img = zeros(capacity, c, h, w, dtype=torch.uint8)
        self.next_frame = zeros(capacity, 3, h, w, dtype=torch.uint8)
        self.inv = zeros(capacity, inv_dim)
        self.next_inv = zeros(capacity, inv_dim)
        self.act = zeros(capacity, dtype=torch.long)
        self.rew = zeros(capacity)
        self.term = zeros(capacity)

    def add(self, img, inv, act, rew, next_img, next_inv, term):
        idx = (self.pos + torch.arange(len(act), device=self.device)) % self.capacity
        to = lambda x: torch.as_tensor(np.asarray(x), device=self.device)
        self.img[idx], self.inv[idx] = to(img), to(inv)
        self.act[idx], self.rew[idx], self.term[idx] = to(act), to(rew), to(term).float()
        self.next_frame[idx], self.next_inv[idx] = to(next_img[:, -3:]), to(next_inv)
        self.pos = (self.pos + len(act)) % self.capacity
        self.size = min(self.size + len(act), self.capacity)

    def sample(self, n):
        idx = torch.randint(0, self.size, (n,), device=self.device)
        img = self.img[idx]
        next_img = torch.cat([img[:, 3:], self.next_frame[idx]], dim=1)
        return (img, self.inv[idx], self.act[idx], self.rew[idx],
                next_img, self.next_inv[idx], self.term[idx])


def main():
    args = parse_args()
    run = args.run_name or f"dqn_s{args.seed}"
    set_seed(args.seed)
    device = torch.device(args.device)

    envs = VecEnv(make_env_fns(args.num_envs, args.seed, **env_kwargs(args)))
    q_net = QNetwork(envs.obs_shape, envs.inv_dim, envs.n_actions, args.use_inventory).to(device)
    target = QNetwork(envs.obs_shape, envs.inv_dim, envs.n_actions, args.use_inventory).to(device)
    target.load_state_dict(q_net.state_dict())
    optim = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, envs.obs_shape, envs.inv_dim, device)
    logger = Logger(os.path.join(args.out_dir, run), vars(args))

    N = args.num_envs
    obs = envs.reset()
    step, updates, update_debt = 0, 0, 0.0
    next_log, next_save, next_sync, t0 = 10_000, args.save_every, args.target_update, time.time()
    losses, q_means = [], []

    while step < args.total_steps:
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end)
                  * step / (args.eps_fraction * args.total_steps))
        with torch.no_grad():
            greedy = q_net(*to_torch(obs, device)).argmax(-1).cpu().numpy()
        explore = np.random.rand(N) < eps
        actions = np.where(explore, np.random.randint(envs.n_actions, size=N), greedy)

        next_obs, reward, term, trunc, infos = envs.step(actions)
        # For finished envs the true successor is the final obs, not the reset obs.
        real_next = {k: v.copy() for k, v in next_obs.items()}
        for i in np.flatnonzero(term | trunc):
            for k in real_next:
                real_next[k][i] = infos[i]["final_obs"][k]
        buffer.add(obs["image"], obs["inv"], actions, reward,
                   real_next["image"], real_next["inv"], term)
        obs = next_obs
        step += N
        logger.add_infos(infos)

        if step >= args.learning_starts:
            update_debt += N * args.replay_ratio
            while update_debt >= 1:
                update_debt -= 1
                img, inv, act, rew, nimg, ninv, done = buffer.sample(args.batch_size)
                with torch.no_grad():
                    # Double DQN: online net picks the action, target net evaluates it.
                    best = q_net(nimg, ninv).argmax(-1, keepdim=True)
                    q_next = target(nimg, ninv).gather(1, best).squeeze(1)
                    y = rew + args.gamma * (1.0 - done) * q_next
                q = q_net(img, inv).gather(1, act[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(q, y)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), args.max_grad_norm)
                optim.step()
                updates += 1
                losses.append(loss.item())
                q_means.append(q.mean().item())
            if step >= next_sync:
                target.load_state_dict(q_net.state_dict())
                next_sync += args.target_update

        if step >= next_log:
            row = logger.write(step, loss=np.mean(losses) if losses else 0.0,
                               q_mean=np.mean(q_means) if q_means else 0.0, epsilon=eps,
                               updates=updates, sps=int(step / (time.time() - t0)))
            losses, q_means = [], []
            if "ep_ret" in row:
                print(f"[{run}] step {step:>9,d}  return {row['ep_ret']:7.3f}  coal {row['ep_coal']:.2f}  "
                      f"iron {row['ep_iron']:.2f}  diamond {row['ep_diamond']:.2f}  "
                      f"dead {row['ep_dead']:.2f}  eps {eps:.2f}  sps {row['sps']}", flush=True)
            next_log += 10_000
        if step >= next_save:
            save_checkpoint(os.path.join(args.ckpt_dir, f"{run}_step{step}.pt"), q_net, args, "dqn")
            next_save += args.save_every

    save_checkpoint(os.path.join(args.ckpt_dir, f"{run}.pt"), q_net, args, "dqn")
    envs.close()


if __name__ == "__main__":
    main()
