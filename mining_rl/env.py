"""Mining task on top of the Crafter open world, plus a small parallel vector env.

The robot is dropped into a procedurally generated world with a set of pickaxes.
It only sees a 64x64 egocentric RGB view (stacked over time) and its own
inventory / health readings, and must find and mine ore while avoiding lava,
zombies and skeletons.
"""
import collections
import functools
import multiprocessing as mp

import crafter
import numpy as np

# Reduced action set: movement (moving also turns the robot to face that way) and mining.
ACTIONS = ["noop", "move_left", "move_right", "move_up", "move_down", "do"]
_CRAFTER_IDS = [crafter.constants.actions.index(a) for a in ACTIONS]

ITEMS = list(crafter.constants.items.keys())          # 16 inventory slots, each 0..9
MINED = ["stone", "coal", "iron", "diamond"]
TOOLS = ["wood_pickaxe", "stone_pickaxe", "iron_pickaxe"]

DEFAULT_REWARD = dict(
    coal=1.0,       # weight per ore block mined
    iron=2.0,
    diamond=5.0,
    stone=0.02,     # tiny bonus for tunnelling, helps exploration
    step=0.001,     # time penalty per step
    damage=0.1,     # penalty per health point lost
    death=1.0,      # extra penalty when the robot is destroyed
)


class MiningEnv:
    """Single mining episode. Observations are dicts with
    ``image``: uint8 (3*frame_stack, 64, 64) and ``inv``: float32 (16,) in [0, 1]."""

    def __init__(self, seed=0, frame_stack=4, max_steps=1000, tools=True,
                 survival=False, reward=None, size=64):
        self._env = crafter.Env(size=(size, size), length=max_steps, seed=seed, reward=False)
        self.frame_stack = frame_stack
        self.tools = tools
        self.survival = survival
        self.reward_cfg = dict(DEFAULT_REWARD, **(reward or {}))
        self.n_actions = len(ACTIONS)
        self.obs_shape = (3 * frame_stack, size, size)
        self.inv_dim = len(ITEMS)
        self._frames = collections.deque(maxlen=frame_stack)

    @property
    def player(self):
        return self._env._player

    def _mined_counts(self):
        # Achievement counters are not capped (inventory saturates at 9).
        ach = self.player.achievements
        return {m: ach[f"collect_{m}"] for m in MINED}

    def _obs(self):
        image = np.concatenate([f.transpose(2, 0, 1) for f in self._frames], axis=0)
        inv = np.array([self.player.inventory[k] for k in ITEMS], np.float32) / 9.0
        return {"image": image, "inv": inv}

    def reset(self):
        frame = self._env.reset()
        if self.tools:
            for tool in TOOLS:
                self.player.inventory[tool] = 1
        self._counts = self._mined_counts()
        self._health = self.player.health
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        self._stats = dict(ret=0.0, length=0, damage=0, **{m: 0 for m in MINED})
        self._stats["lava"] = 0.0
        return self._obs()

    def step(self, action):
        cfg, player = self.reward_cfg, self.player
        if not self.survival:
            # A robot does not get hungry, thirsty or tired.
            player._hunger = player._thirst = player._fatigue = 0
        frame, _, done, _ = self._env.step(_CRAFTER_IDS[int(action)])
        self._frames.append(frame)

        counts = self._mined_counts()
        gained = {m: counts[m] - self._counts[m] for m in MINED}
        self._counts = counts
        lost = max(0, self._health - player.health)
        self._health = player.health
        dead = player.health <= 0

        reward = sum(cfg[m] * gained[m] for m in MINED)
        reward -= cfg["step"] + cfg["damage"] * lost + cfg["death"] * dead

        s = self._stats
        s["ret"] += reward
        s["length"] += 1
        s["damage"] += lost
        for m in MINED:
            s[m] += gained[m]

        terminated = bool(dead)
        truncated = bool(done and not dead)
        info = {}
        if terminated or truncated:
            # Death cause: walking into lava is instant, everything else is mob damage.
            s["lava"] = float(dead and self._env._world[player.pos][0] == "lava")
            info["episode"] = dict(s, dead=float(dead))
        return self._obs(), float(reward), terminated, truncated, info

    def render(self, size=256):
        return self._env.render((size, size))


def make_env_fns(n, seed, **env_kwargs):
    """Picklable constructors; each worker gets its own stream of worlds."""
    return [functools.partial(MiningEnv, seed=seed * 1000 + i, **env_kwargs) for i in range(n)]


def _worker(remote, env_fn):
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, term, trunc, info = env.step(data)
            if term or trunc:
                info["final_obs"] = obs
                obs = env.reset()
            remote.send((obs, reward, term, trunc, info))
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "close":
            remote.close()
            break


def _stack(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


class VecEnv:
    """Runs environments in subprocesses and auto-resets finished ones in the same step.
    The last observation of a finished episode is returned in ``info['final_obs']``."""

    def __init__(self, env_fns):
        ctx = mp.get_context("spawn")
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
        self.procs = []
        for work_remote, fn in zip(work_remotes, env_fns):
            p = ctx.Process(target=_worker, args=(work_remote, fn), daemon=True)
            p.start()
            work_remote.close()
            self.procs.append(p)
        probe = env_fns[0]()
        self.num_envs = len(env_fns)
        self.n_actions, self.obs_shape, self.inv_dim = probe.n_actions, probe.obs_shape, probe.inv_dim

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return _stack([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        obs, rews, terms, truncs, infos = zip(*[r.recv() for r in self.remotes])
        return (_stack(obs), np.array(rews, np.float32), np.array(terms),
                np.array(truncs), list(infos))

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
