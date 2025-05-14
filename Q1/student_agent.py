import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

HIDDEN     = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------

def make_env():
    env = gym.make("Pendulum-v1")
    return env

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(), # Using Tanh activation often works well
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
        )
        self.mu = nn.Linear(HIDDEN, act_dim)
        self.action_scale = 2.0
        self.log_sigma = nn.Parameter(torch.zeros(act_dim)) # Learnable log std dev

    def forward(self, obs):
        x = self.fc(obs)
        mu = torch.tanh(self.mu(x)) * self.action_scale # Scale action to [-2, 2]
        sigma = torch.exp(self.log_sigma) # Use fixed learned std dev
        sigma = sigma.expand_as(mu)
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.Tanh(), # Using Tanh activation
            nn.Linear(HIDDEN, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, obs):
        return self.fc(obs)

class PPO_Agent:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(DEVICE)
        self.critic = Critic(obs_dim).to(DEVICE)

    def act(self, obs):
        if isinstance(obs, np.ndarray):
             obs = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE) # Add batch dim
        else:
             obs = obs.to(DEVICE)
             if obs.ndim == 1: # Ensure batch dimension if single obs passed
                 obs = obs.unsqueeze(0)


        with torch.no_grad():
            mu, sigma = self.actor(obs)
            value = self.critic(obs)

        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1) # Sum across action dimensions if act_dim > 1

        action_clipped = torch.clamp(action, -self.actor.action_scale, self.actor.action_scale)

        return action_clipped.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy()

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.agent = PPO_Agent(3, 1) # obs_dim=3 for Pendulum, act_dim=1 for action space
        self.agent.actor.load_state_dict(torch.load("models/ppo_actor_ep20000.pth"))
        self.agent.critic.load_state_dict(torch.load("models/ppo_critic_ep20000.pth"))
        self.agent.actor.eval()
        self.agent.critic.eval()

    def act(self, observation):
        """Return action based on observation."""
        action, _, _ = self.agent.act(observation)
        return action