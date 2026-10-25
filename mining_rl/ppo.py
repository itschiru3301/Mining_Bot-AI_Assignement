"""Proximal Policy Optimisation (clipped surrogate + GAE) for the mining task.

Usage:  python -m mining_rl.ppo --seed 1 --total-steps 5000000
"""
import argparse
import os
import time

import numpy as np
import torch

from mining_rl.common import Logger, env_kwargs, save_checkpoint, set_seed, to_torch
from mining_rl.env import VecEnv, make_env_fns
from mining_rl.networks import ActorCritic


def parse_args():
    p = argparse.ArgumentParser(description="Train a PPO mining agent")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # environment / ablations
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--no-tools", dest="tools", action="store_false")
    p.add_argument("--survival", action="store_true", help="keep hunger/thirst/fatigue on")
    p.add_argument("--no-inventory", dest="use_inventory", action="store_false")
    p.add_argument("--no-damage-penalty", action="store_true")
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """dones[t] marks that obs t starts a new episode (the previous step ended one)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            alive, v_next = 1.0 - next_done, next_value
        else:
            alive, v_next = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * v_next * alive - values[t]
        running = delta + gamma * lam * alive * running
        adv[t] = running
    return adv, adv + values


def main():
    args = parse_args()
    run = args.run_name or f"ppo_s{args.seed}"
    set_seed(args.seed)
    device = torch.device(args.device)

    envs = VecEnv(make_env_fns(args.num_envs, args.seed, **env_kwargs(args)))
    model = ActorCritic(envs.obs_shape, envs.inv_dim, envs.n_actions, args.use_inventory).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    logger = Logger(os.path.join(args.out_dir, run), vars(args))

    T, N = args.num_steps, args.num_envs
    batch = T * N
    mb_size = batch // args.minibatches
    img_buf = torch.zeros((T, N) + envs.obs_shape, dtype=torch.uint8, device=device)
    inv_buf = torch.zeros((T, N, envs.inv_dim), device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros((T, N), device=device) for _ in range(4))

    img, inv = to_torch(envs.reset(), device)
    done = torch.zeros(N, device=device)
    n_updates = args.total_steps // batch
    step, next_save, t0 = 0, args.save_every, time.time()

    for update in range(1, n_updates + 1):
        for group in optim.param_groups:
            group["lr"] = args.lr * (1.0 - (update - 1) / n_updates)

        # ---- collect a rollout ----
        for t in range(T):
            img_buf[t], inv_buf[t], done_buf[t] = img, inv, done
            with torch.no_grad():
                action, logp, _, value = model.act(img, inv)
            act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

            obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            step += N
            # A time limit is not a real terminal state: bootstrap from V(final obs).
            for i in np.flatnonzero(trunc):
                fin = infos[i]["final_obs"]
                with torch.no_grad():
                    _, v_fin = model(*to_torch({k: v[None] for k, v in fin.items()}, device))
                reward[i] += args.gamma * v_fin.item()
            rew_buf[t] = torch.as_tensor(reward, device=device)
            img, inv = to_torch(obs, device)
            done = torch.as_tensor(term | trunc, dtype=torch.float32, device=device)
            logger.add_infos(infos)

        with torch.no_grad():
            _, next_value = model(img, inv)
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, next_value, done,
                               args.gamma, args.gae_lambda)

        # ---- optimise the clipped surrogate ----
        b_img = img_buf.reshape((batch,) + envs.obs_shape)
        b_inv = inv_buf.reshape(batch, -1)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for start in range(0, batch, mb_size):
                j = perm[start:start + mb_size]
                _, logp, entropy, value = model.act(b_img[j], b_inv[j], b_act[j])
                log_ratio = logp - b_logp[j]
                ratio = log_ratio.exp()
                a = (b_adv[j] - b_adv[j].mean()) / (b_adv[j].std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * (value - b_ret[j]).pow(2).mean()
                ent = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean()
                    clipfrac = ((ratio - 1).abs() > args.clip).float().mean()
                stats.append([pg_loss.item(), v_loss.item(), ent.item(), kl.item(), clipfrac.item()])

        pg, vl, en, kl, cf = np.mean(stats, axis=0)
        row = logger.write(step, pg_loss=pg, v_loss=vl, entropy=en, approx_kl=kl, clipfrac=cf,
                           sps=int(step / (time.time() - t0)))
        if "ep_ret" in row:
            print(f"[{run}] step {step:>9,d}  return {row['ep_ret']:7.3f}  coal {row['ep_coal']:.2f}  "
                  f"iron {row['ep_iron']:.2f}  diamond {row['ep_diamond']:.2f}  "
                  f"dead {row['ep_dead']:.2f}  ent {en:.2f}  sps {row['sps']}", flush=True)
        if step >= next_save:
            save_checkpoint(os.path.join(args.ckpt_dir, f"{run}_step{step}.pt"), model, args, "ppo")
            next_save += args.save_every

    save_checkpoint(os.path.join(args.ckpt_dir, f"{run}.pt"), model, args, "ppo")
    envs.close()


if __name__ == "__main__":
    main()
