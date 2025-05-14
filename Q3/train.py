import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import copy
import time
import os

def make_dmc_env(
    env_name: str, seed: int, frame_stack: int = 1, flatten: bool = True, 
    use_pixels: bool = False, width: int = 84, height: int = 84, 
) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    dmc_env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
    env = DmControltoGymnasium(
        dmc_env,
        render_mode="rgb_array" if use_pixels else None,
        render_kwargs={"width": width, "height": height, "camera_id": 0} if use_pixels else None,
    )
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)
    if use_pixels:
        pass 

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6 

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space_low, action_space_high,
                 hidden_dims=[512, 512, 256], log_std_init=0.0):
        super(Actor, self).__init__()
        
        self.action_scale = torch.tensor((action_space_high - action_space_low) / 2., dtype=torch.float32)
        self.action_bias = torch.tensor((action_space_high + action_space_low) / 2., dtype=torch.float32)

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)

    def forward(self, state):
        x = self.trunk(state)
        mean = self.mean_linear(x)
        log_std_clamped = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std_broadcast = log_std_clamped.unsqueeze(0).expand_as(mean)
        return mean, log_std_broadcast

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state) 
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            u = mean 
        else:
            u = normal.rsample() 

        action_tanh = torch.tanh(u) 
        
        log_prob_u = normal.log_prob(u).sum(dim=1, keepdim=True) 
        
        tanh_correction = torch.log(1 - action_tanh.pow(2) + EPS).sum(dim=1, keepdim=True) 
        
        log_prob_squashed_action = log_prob_u - tanh_correction 
        
        action_env = action_tanh * self.action_scale.to(state.device) + self.action_bias.to(state.device)
        
        return action_env, log_prob_squashed_action, action_tanh


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 256]):
        super(Critic, self).__init__()
        def create_q_network(h_dims):
            layers = []
            prev_d = state_dim + action_dim
            for h_dim in h_dims:
                layers.append(nn.Linear(prev_d, h_dim))
                layers.append(nn.ReLU())
                prev_d = h_dim
            layers.append(nn.Linear(prev_d, 1))
            return nn.Sequential(*layers)
        self.q1_net = create_q_network(hidden_dims)
        self.q2_net = create_q_network(hidden_dims)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        q1 = self.q1_net(xu) # Shape [batch_size, 1]
        q2 = self.q2_net(xu) # Shape [batch_size, 1]
        return q1, q2

class SACAgent:
    def __init__(self, state_dim, action_space,
                 lr=3e-4,                  
                 gamma=0.99,               
                 tau=0.005,                
                 alpha_init=1.0,           
                 buffer_capacity=1_000_000,
                 batch_size=1024,          
                 hidden_dims_actor=[512, 512, 256], 
                 hidden_dims_critic=[512, 512, 256],
                 log_std_init=0.0,         
                 auto_entropy_tuning=True,
                 target_entropy_scale=1.0,
                 device='cuda',
                 grad_clip_norm=None):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm

        action_dim = action_space.shape[0]
        action_space_low = action_space.low
        action_space_high = action_space.high

        self.actor = Actor(state_dim, action_dim, action_space_low, action_space_high,
                           hidden_dims_actor, log_std_init).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, hidden_dims_critic).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() * target_entropy_scale
            self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr) # Using same lr as actor/critic for alpha
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        self.actor.action_scale = self.actor.action_scale.to(self.device)
        self.actor.action_bias = self.actor.action_bias.to(self.device)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state.astype(np.float32)).unsqueeze(0).to(self.device)
        action_env, _, _ = self.actor.sample(state_tensor, deterministic=evaluate)
        return action_env.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state.astype(np.float32),
                               action.astype(np.float32),
                               float(reward),
                               next_state.astype(np.float32),
                               int(done))

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done_mask = torch.FloatTensor(1.0 - done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action_env, next_log_prob_tanh, _ = self.actor.sample(next_state)
            q1_next_target, q2_next_target = self.critic_target(next_state, next_action_env)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) # Shape [batch_size, 1]
            next_q_value = reward + done_mask * self.gamma * (min_q_next_target - self.alpha * next_log_prob_tanh)

        current_q1, current_q2 = self.critic(state, action) # Shapes [batch_size, 1]
        critic_loss_q1 = F.mse_loss(current_q1, next_q_value)
        critic_loss_q2 = F.mse_loss(current_q2, next_q_value)
        critic_loss = critic_loss_q1 + critic_loss_q2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        pi_action_env, pi_log_prob_tanh, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi_action_env)
        min_q_pi = torch.min(q1_pi, q2_pi) # Shape [batch_size, 1]

        actor_loss = (self.alpha * pi_log_prob_tanh - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (pi_log_prob_tanh + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), self.alpha

    def save_models(self, env_name, suffix="", actor_path=None, critic_path=None, alpha_path=None):
        if not os.path.exists('models/'): os.makedirs('models/')
        actor_p = actor_path or f"models/sac_actor_{env_name}{suffix}.pth"
        critic_p = critic_path or f"models/sac_critic_{env_name}{suffix}.pth"
        print(f'Saving models to {actor_p}, {critic_p}', end='')
        torch.save(self.actor.state_dict(), actor_p)
        torch.save(self.critic.state_dict(), critic_p)
        if self.auto_entropy_tuning:
            alpha_p = alpha_path or f"models/sac_log_alpha_{env_name}{suffix}.pth"
            torch.save(self.log_alpha, alpha_p); print(f', {alpha_p}')
        else: print()

    def load_models(self, actor_path, critic_path, alpha_path=None):
        print(f'Loading models from {actor_path}, {critic_path}', end='')
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters(): param.requires_grad = False
        if self.auto_entropy_tuning and alpha_path:
            self.log_alpha = torch.load(alpha_path, map_location=self.device)
            if not self.log_alpha.requires_grad: self.log_alpha.requires_grad_(True)
            self.alpha = self.log_alpha.exp().item()
            alpha_lr = self.alpha_optimizer.defaults['lr'] if self.alpha_optimizer else 3e-4 # Fallback lr
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            print(f', {alpha_path}')
        else:
            print()
        self.actor.action_scale = self.actor.action_scale.to(self.device)
        self.actor.action_bias = self.actor.action_bias.to(self.device)

def train():
    ENV_NAME = "humanoid-walk"
    SEED = 42
    USE_PIXELS = False
    FRAME_STACK = 1 

    LR = 3e-4                 
    GAMMA = 0.99                
    TAU = 0.005                 
    BUFFER_CAPACITY = 1_000_000 
    BATCH_SIZE = 1024           

    HIDDEN_DIMS_ACTOR = [512, 512, 256]
    HIDDEN_DIMS_CRITIC = [512, 512, 256]
    LOG_STD_INIT = 0.0          

    AUTO_ENTROPY = True
    INITIAL_ALPHA = 1.0         
    TARGET_ENTROPY_SCALE = 1.0

    MAX_TIMESTEPS = int(5e6)    
    LEARNING_STARTS = BATCH_SIZE

    UPDATES_PER_STEP = 1        

    EVAL_FREQ = 20_000          
    N_EVAL_EPISODES = 5         
    SAVE_FREQ = 100_000         

    GRAD_CLIP_NORM = None       

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    env = make_dmc_env(ENV_NAME, SEED, frame_stack=FRAME_STACK, use_pixels=USE_PIXELS, flatten=True)
    eval_env = make_dmc_env(ENV_NAME, SEED + 999, frame_stack=FRAME_STACK, use_pixels=USE_PIXELS, flatten=True) 

    state_dim = env.observation_space.shape[0]
    action_space = env.action_space

    agent = SACAgent(state_dim, action_space,
                     lr=LR, gamma=GAMMA, tau=TAU, alpha_init=INITIAL_ALPHA,
                     buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE,
                     hidden_dims_actor=HIDDEN_DIMS_ACTOR, hidden_dims_critic=HIDDEN_DIMS_CRITIC,
                     log_std_init=LOG_STD_INIT,
                     auto_entropy_tuning=AUTO_ENTROPY, target_entropy_scale=TARGET_ENTROPY_SCALE,
                     device=DEVICE, grad_clip_norm=GRAD_CLIP_NORM)

    episode_rewards_deque = deque(maxlen=100)
    total_timesteps = 0
    episode_num = 0

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0

    print(f"Starting training for {ENV_NAME} (State Vectors, SB3-Aligned Params) on {DEVICE}")
    print(f"Obs dim: {state_dim}, Action dim: {action_space.shape[0]}, Batch Size: {BATCH_SIZE}")
    print(f"Target Entropy: {agent.target_entropy if agent.auto_entropy_tuning else 'N/A (fixed alpha)'}, Initial Alpha: {INITIAL_ALPHA}")

    for t_step in range(1, MAX_TIMESTEPS + 1):
        total_timesteps = t_step

        if total_timesteps < LEARNING_STARTS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_timesteps += 1

        if total_timesteps >= LEARNING_STARTS and total_timesteps % UPDATES_PER_STEP == 0:
            loss_info = agent.update()
            # if loss_info and total_timesteps % 2000 == 0: # Optional: Log losses less frequently
            #     print(f"T: {total_timesteps}, C_loss: {loss_info[0]:.3f}, A_loss: {loss_info[1]:.3f}, Alpha: {loss_info[2]:.3f}")


        if done:
            episode_rewards_deque.append(episode_reward)
            avg_ep_reward = np.mean(episode_rewards_deque) if episode_rewards_deque else 0.0
            if episode_num % 20 == 0 or total_timesteps < LEARNING_STARTS + 5000 :
                 print(f"T: {total_timesteps} Ep: {episode_num+1} Steps: {episode_timesteps} R: {episode_reward:.2f} AvgR(100): {avg_ep_reward:.2f} Alpha: {agent.alpha:.3f}")

            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if total_timesteps % EVAL_FREQ == 0 and total_timesteps >= LEARNING_STARTS:
            eval_avg_reward = 0.
            for _ in range(N_EVAL_EPISODES):
                eval_state, _ = eval_env.reset()
                eval_done = False
                eval_ep_r = 0
                eval_ep_s = 0
                max_ep_len = eval_env.spec.max_episode_steps if eval_env.spec and eval_env.spec.max_episode_steps else 1000

                while not eval_done and eval_ep_s < max_ep_len:
                    eval_action = agent.select_action(eval_state, evaluate=True)
                    eval_state_new, r, term, trunc, _ = eval_env.step(eval_action)
                    eval_state = eval_state_new # Update eval_state
                    eval_done = term or trunc
                    eval_ep_r += r
                    eval_ep_s +=1
                eval_avg_reward += eval_ep_r
            eval_avg_reward /= N_EVAL_EPISODES
            print("---------------------------------------")
            print(f"Evaluation over {N_EVAL_EPISODES} episodes: {eval_avg_reward:.3f} at timestep {total_timesteps}")
            print("---------------------------------------")

        if total_timesteps % SAVE_FREQ == 0 and total_timesteps >= LEARNING_STARTS:
            agent.save_models(ENV_NAME, suffix=f"_state_sb3aligned_ts{total_timesteps}")

    agent.save_models(ENV_NAME, suffix="_state_sb3aligned_final")
    env.close()
    eval_env.close()
    print("Training finished.")

if __name__ == "__main__":
    train()