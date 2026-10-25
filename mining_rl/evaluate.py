"""Evaluate trained agents (or a uniform-random baseline) on held-out worlds.

Usage:
  python -m mining_rl.evaluate --agent random --episodes 100
  python -m mining_rl.evaluate --checkpoint checkpoints/ppo_s1.pt --episodes 100
"""
import argparse
import json
import os

import numpy as np
import torch

from mining_rl.common import env_kwargs, to_torch
from mining_rl.env import VecEnv, make_env_fns
from mining_rl.networks import ActorCritic, QNetwork

EVAL_SEED = 777  # worlds never seen in training (training uses seeds 1..3)


def load_agent(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = argparse.Namespace(**ckpt["config"])
    return ckpt["algo"], cfg, ckpt["model"]


def build_policy(algo, cfg, state, envs, device, greedy, eps):
    if algo == "random":
        return lambda obs: np.random.randint(envs.n_actions, size=envs.num_envs)
    net_cls = ActorCritic if algo == "ppo" else QNetwork
    net = net_cls(envs.obs_shape, envs.inv_dim, envs.n_actions, cfg.use_inventory).to(device)
    net.load_state_dict(state)
    net.eval()

    @torch.no_grad()
    def policy(obs):
        img, inv = to_torch(obs, device)
        if algo == "ppo":
            return net.act(img, inv, greedy=greedy)[0].cpu().numpy()
        actions = net(img, inv).argmax(-1).cpu().numpy()
        explore = np.random.rand(len(actions)) < eps
        return np.where(explore, np.random.randint(envs.n_actions, size=len(actions)), actions)

    return policy


def evaluate(policy, envs, episodes):
    results, obs = [], envs.reset()
    while len(results) < episodes:
        obs, _, _, _, infos = envs.step(policy(obs))
        results += [info["episode"] for info in infos if "episode" in info]
    return results[:episodes]


def main():
    p = argparse.ArgumentParser(description="Evaluate a mining agent")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--agent", default=None, help="'random' for the random baseline")
    p.add_argument("--name", default=None, help="label used in the output file")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--greedy", action="store_true", help="PPO: take argmax instead of sampling")
    p.add_argument("--eps", type=float, default=0.01, help="DQN: evaluation epsilon")
    p.add_argument("--out-dir", default="results/eval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.agent == "random":
        algo, state = "random", None
        cfg = argparse.Namespace(frame_stack=4, max_steps=1000, tools=True, survival=False)
    else:
        algo, cfg, state = load_agent(args.checkpoint, device)
    name = args.name or (algo if algo == "random" else os.path.splitext(os.path.basename(args.checkpoint))[0])

    # Evaluate everyone on the same reward (the damage-penalty ablation is only a training change).
    cfg.no_damage_penalty = False
    envs = VecEnv(make_env_fns(args.num_envs, EVAL_SEED, **env_kwargs(cfg)))
    np.random.seed(EVAL_SEED)
    policy = build_policy(algo, cfg, state, envs, device, args.greedy, args.eps)
    episodes = evaluate(policy, envs, args.episodes)
    envs.close()

    summary = {k: float(np.mean([e[k] for e in episodes])) for k in episodes[0]}
    summary.update({f"{k}_std": float(np.std([e[k] for e in episodes])) for k in episodes[0]})
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump({"name": name, "algo": algo, "summary": summary, "episodes": episodes}, f, indent=1)
    print(f"{name:>16s}: return {summary['ret']:6.2f}  coal {summary['coal']:.2f}  iron {summary['iron']:.2f}  "
          f"diamond {summary['diamond']:.2f}  stone {summary['stone']:.1f}  damage {summary['damage']:.1f}  "
          f"died {summary['dead']:.0%}  length {summary['length']:.0f}")


if __name__ == "__main__":
    main()
