"""Record a GIF of a trained agent mining.

Usage:  python -m mining_rl.record --checkpoint checkpoints/ppo_s1.pt --out results/media/ppo.gif
"""
import argparse
import os

import imageio
import numpy as np
import torch

from mining_rl.common import env_kwargs, to_torch
from mining_rl.env import MiningEnv
from mining_rl.evaluate import EVAL_SEED, load_agent
from mining_rl.networks import ActorCritic, QNetwork


def main():
    p = argparse.ArgumentParser(description="Record an episode of a mining agent")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="results/media/agent.gif")
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    args = p.parse_args()

    torch.manual_seed(args.seed)  # PPO samples its actions; fix the sampling for repeatable GIFs
    algo, cfg, state = load_agent(args.checkpoint, "cpu")
    cfg.no_damage_penalty = False
    env = MiningEnv(seed=args.seed, **env_kwargs(cfg))
    net = (ActorCritic if algo == "ppo" else QNetwork)(env.obs_shape, env.inv_dim, env.n_actions, cfg.use_inventory)
    net.load_state_dict(state)
    net.eval()

    obs, frames, total = env.reset(), [env.render(args.size)], 0.0
    for _ in range(args.max_frames):
        with torch.no_grad():
            batch = to_torch({k: v[None] for k, v in obs.items()}, "cpu")
            action = net.act(*batch)[0] if algo == "ppo" else net(*batch).argmax(-1)
        obs, reward, term, trunc, info = env.step(int(action))
        total += reward
        frames.append(env.render(args.size))
        if term or trunc:
            break
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, [np.asarray(f) for f in frames], duration=1000 / args.fps, loop=0)
    print(f"saved {len(frames)} frames to {args.out}  (return {total:.2f}, mined {env._stats})")


if __name__ == "__main__":
    main()
