import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import time # For potential rendering delay
import matplotlib.pyplot as plt     

# -------- hyper-parameters --------
GAMMA      = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS   = 0.2
LR         = 1e-4 # Might need adjustment (e.g., 3e-4)
K_EPOCHS   = 10
T_HORIZON  = 2048 # Number of steps to collect per update cycle (replaces implicit buffer size)
MINIBATCH_SIZE = 64 # Renamed BATCH_SIZE for clarity
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
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LR)

        self.buffer = [] # Store (obs, action, log_prob, reward, done, value) tuples

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

    def store(self, obs, action, log_prob, reward, done, value):
        self.buffer.append((obs, action, log_prob, reward, done, value))

    def _compute_gae(self, rewards, values, dones, last_value, last_done):
        """ Computes Generalized Advantage Estimation (GAE). """
        num_steps = len(rewards) # Should be the buffer size (e.g., <= T_HORIZON)

        advantages = torch.zeros(num_steps, dtype=torch.float32, device=DEVICE)

        last_gae_lam = 0.0 # Use float for consistency

        last_value_tensor = torch.tensor(last_value, dtype=torch.float32, device=DEVICE).squeeze()
        next_non_terminal = (1.0 - torch.tensor(last_done, dtype=torch.float32, device=DEVICE)).squeeze()

        if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
            print(f"Warning: Incorrect dimensions for inputs to GAE! "
                  f"rewards: {rewards.shape}, values: {values.shape}, dones: {dones.shape}")
            return torch.zeros_like(rewards), torch.zeros_like(values)


        for t in reversed(range(num_steps)):
            current_value_t = values[t].squeeze() # Squeeze potential trailing dim
            reward_t = rewards[t].squeeze()
            done_t_plus_1 = dones[t+1].squeeze() if t < num_steps - 1 else torch.tensor(last_done, dtype=torch.float32, device=DEVICE).squeeze() # Use last_done for the edge case

            if t == num_steps - 1:
                next_value = last_value_tensor       # Should be 0-dim tensor
                next_non_term = next_non_terminal # Should be 0-dim tensor
            else:
                next_value = values[t+1].squeeze() # Ensure 0-dim tensor
                next_non_term = (1.0 - done_t_plus_1).squeeze() # Ensure 0-dim tensor

            delta = reward_t + GAMMA * next_value * next_non_term - current_value_t
            last_gae_lam = delta + GAMMA * LAMBDA_GAE * next_non_term * last_gae_lam
            advantages[t] = last_gae_lam # Assign scalar result to element t

        returns = advantages + values

        if advantages.ndim != 1 or returns.ndim != 1:
             print(f"Error in GAE: Output dimensions are wrong! "
                   f"advantages: {advantages.shape}, returns: {returns.shape}")

        return advantages, returns

    def update(self, last_value, last_done):
        if not self.buffer:
            print("Warning: Update called with empty buffer.")
            return

        obs_list, action_list, old_log_prob_list, reward_list, done_list, old_value_list = zip(*self.buffer)

        obs_np = np.stack(obs_list)
        actions_np = np.stack(action_list)
        old_log_probs_np = np.array(old_log_prob_list, dtype=np.float32)
        rewards_np = np.array(reward_list, dtype=np.float32)
        dones_np = np.array(done_list, dtype=np.float32)
        old_values_np = np.array(old_value_list, dtype=np.float32) # Shape [num_steps, 1]

        obs = torch.from_numpy(obs_np).float().to(DEVICE)
        actions = torch.from_numpy(actions_np).float().to(DEVICE)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(DEVICE)
        rewards = torch.from_numpy(rewards_np).to(DEVICE)
        dones = torch.from_numpy(dones_np).to(DEVICE)
        old_values = torch.from_numpy(old_values_np).to(DEVICE) # Shape [num_steps, 1]

        old_values = old_values.squeeze(-1) # Now shape [num_steps]

        advantages, returns = self._compute_gae(rewards, old_values, dones, last_value, last_done)


        if advantages.numel() > 1 and advantages.std() > 1e-8:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        elif advantages.numel() == 1:
             advantages = advantages - advantages.mean() # Just center if only one element

        num_samples = len(self.buffer)
        indices = np.arange(num_samples)

        for _ in range(K_EPOCHS):
            np.random.shuffle(indices) # Shuffle indices for each epoch
            for start in range(0, num_samples, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices] # Should slice correctly from 1D returns

                mu, sigma = self.actor(batch_obs)
                dist = torch.distributions.Normal(mu, sigma)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                current_values = self.critic(batch_obs).squeeze(-1) # Input: [batch, 1], Target: [batch] -> Correct
                critic_loss = nn.MSELoss()(current_values, batch_returns)

                entropy = dist.entropy().mean()
                entropy_loss = -0.01 * entropy # Adjust coefficient if needed

                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.optimizer.step()

        self.buffer.clear()

def train(num_episodes=1000, render=False):
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}, action_scale: {env.action_space.high[0]}")
    print(f"Device: {DEVICE}")

    total_steps      = 0
    recent_rewards   = deque(maxlen=100)        # 最近 100 集平均
    all_rewards      = []                       # ✔ NEW: 完整曲線用

    for ep in range(1, num_episodes + 1):
        obs, _     = env.reset(seed=np.random.randint(1_000_000))
        done       = False
        truncated  = False
        ep_reward  = 0.0

        while not (done or truncated):
            if render and ep % 50 == 0:
                try:
                    env.render()
                except Exception as e:
                    print(f"Render failed: {e}")
                    render = False

            action, logp, val = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, logp, reward, terminated, val)
            obs        = next_obs
            ep_reward += reward
            total_steps += 1

            buffer_full = len(agent.buffer) >= T_HORIZON
            if buffer_full or done:
                with torch.no_grad():
                    last_v = agent.critic(torch.from_numpy(obs).float()
                                           .unsqueeze(0).to(DEVICE)).squeeze().cpu().item()
                    last_v *= (1 - terminated)            # bootstrap 但終局不用
                agent.update(last_v, terminated)

        recent_rewards.append(ep_reward)
        all_rewards.append(ep_reward)           # ✔ NEW
        avg100 = np.mean(recent_rewards)

        if ep % 5000 == 0:
            torch.save(agent.actor.state_dict(), f"models/ppo_actor_ep{ep}.pth")
            torch.save(agent.critic.state_dict(), f"models/ppo_critic_ep{ep}.pth")
            print(f">>> checkpoint saved @ episode {ep}")

            plt.figure(figsize=(10,4))
            plt.plot(all_rewards, label='Episode reward')
            if len(all_rewards) >= 100:
                kernel = np.ones(100) / 100
                smoothed = np.convolve(all_rewards, kernel, mode='valid')
                plt.plot(range(99, len(all_rewards)), smoothed,
                        linewidth=2, label='Moving avg (100)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('PPO on Pendulum-v1')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'training_curve_ep{ep}.png')
            plt.show()
            print("圖已存檔：training_curve.png")
        
        if ep % 10 == 0:
            print(f"Ep {ep:5d} | Steps {total_steps:8d} | "
                  f"EpR {ep_reward:8.1f} | Avg100 {avg100:8.1f}")

    env.close()

    

if __name__ == "__main__":
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPO_Agent(obs_dim, act_dim)
    train(num_episodes=100000, render=False) # Pendulum converges relatively fast
    
    # save the model
    torch.save(agent.actor.state_dict(), "ppo_actor.pth")
    torch.save(agent.critic.state_dict(), "ppo_critic.pth")
    print("Model saved.")