"""Small helpers shared by the training scripts: seeding, logging, checkpoints."""
import csv
import json
import os
import random
import time

import numpy as np
import torch

ENV_KEYS = ("frame_stack", "max_steps", "tools", "survival")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_torch(obs, device):
    return (torch.as_tensor(obs["image"], device=device),
            torch.as_tensor(obs["inv"], device=device))


def env_kwargs(args):
    kw = {k: getattr(args, k) for k in ENV_KEYS}
    if getattr(args, "no_damage_penalty", False):
        kw["reward"] = dict(damage=0.0, death=0.0)
    return kw


class Logger:
    """Collects finished-episode stats and writes one CSV row per log interval."""

    def __init__(self, run_dir, config):
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        self.path = os.path.join(run_dir, "progress.csv")
        self.episodes = []
        self.writer = None
        self.start = time.time()

    def add_infos(self, infos):
        for info in infos:
            if "episode" in info:
                self.episodes.append(info["episode"])

    def write(self, step, **extra):
        row = {"step": step, "time": round(time.time() - self.start, 1), "episodes": len(self.episodes)}
        if self.episodes:
            for k in self.episodes[0]:
                row[f"ep_{k}"] = float(np.mean([e[k] for e in self.episodes]))
        row.update(extra)
        if not self.episodes:  # nothing finished yet; keep the CSV schema fixed
            return row
        if self.writer is None:
            self._file = open(self.path, "w", newline="")
            self.writer = csv.DictWriter(self._file, fieldnames=list(row.keys()), extrasaction="ignore")
            self.writer.writeheader()
        self.writer.writerow(row)
        self._file.flush()
        self.episodes = []
        return row


def save_checkpoint(path, model, args, algo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"algo": algo, "config": vars(args), "model": model.state_dict()}, path)
