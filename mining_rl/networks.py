"""Neural networks shared by the PPO and DQN agents."""
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, gain=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class Encoder(nn.Module):
    """CNN over stacked RGB frames, fused with an MLP over the inventory vector."""

    def __init__(self, obs_shape, inv_dim, feat_dim=512, use_inventory=True):
        super().__init__()
        c, h, w = obs_shape
        self.use_inventory = use_inventory
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        n_inv = 64 if use_inventory else 0
        if use_inventory:
            self.inv_mlp = nn.Sequential(layer_init(nn.Linear(inv_dim, n_inv)), nn.ReLU())
        self.fuse = nn.Sequential(layer_init(nn.Linear(n_flat + n_inv, feat_dim)), nn.ReLU())

    def forward(self, image, inv):
        x = self.cnn(image.float() / 255.0)
        if self.use_inventory:
            x = torch.cat([x, self.inv_mlp(inv)], dim=1)
        return self.fuse(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, inv_dim, n_actions, use_inventory=True):
        super().__init__()
        self.encoder = Encoder(obs_shape, inv_dim, use_inventory=use_inventory)
        self.actor = layer_init(nn.Linear(512, n_actions), gain=0.01)
        self.critic = layer_init(nn.Linear(512, 1), gain=1.0)

    def forward(self, image, inv):
        z = self.encoder(image, inv)
        return self.actor(z), self.critic(z).squeeze(-1)

    def act(self, image, inv, action=None, greedy=False):
        logits, value = self(image, inv)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class QNetwork(nn.Module):
    def __init__(self, obs_shape, inv_dim, n_actions, use_inventory=True):
        super().__init__()
        self.encoder = Encoder(obs_shape, inv_dim, use_inventory=use_inventory)
        self.head = layer_init(nn.Linear(512, n_actions), gain=1.0)

    def forward(self, image, inv):
        return self.head(self.encoder(image, inv))
