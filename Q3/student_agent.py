import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque 
from dmc import make_dmc_env 

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class Actor(nn.Module):
    def __init__(self, stacked_state_dim, action_dim, action_space_low, action_space_high,
                 hidden_dims=[512, 512, 256], log_std_init=0.0):
        super(Actor, self).__init__()
        self.action_scale = torch.tensor((action_space_high - action_space_low) / 2., dtype=torch.float32)
        self.action_bias = torch.tensor((action_space_high + action_space_low) / 2., dtype=torch.float32)

        layers = []
        prev_dim = stacked_state_dim 
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init, requires_grad=True)

    def forward(self, stacked_state): 
        x = self.trunk(stacked_state)
        mean = self.mean_linear(x)
        log_std_clamped = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std_broadcast = log_std_clamped.unsqueeze(0).expand_as(mean)
        return mean, log_std_broadcast

    def sample(self, stacked_state, deterministic=False): 
        mean, log_std = self.forward(stacked_state)
        std = log_std.exp()
        normal = Normal(mean, std)
        if deterministic: u = mean
        else: u = normal.rsample() 
        action_tanh = torch.tanh(u)
        action_env = action_tanh * self.action_scale.to(stacked_state.device) + self.action_bias.to(stacked_state.device)
        return action_env, None, action_tanh 

class TestSACAgent: 
    def __init__(self, stacked_state_dim, action_space,
                 hidden_dims_actor=[512, 512, 256],
                 log_std_init=0.0,
                 device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"TestAgent using device: {self.device}")

        action_dim = action_space.shape[0]
        action_space_low = action_space.low
        action_space_high = action_space.high

        self.actor = Actor(stacked_state_dim, action_dim, action_space_low, action_space_high,
                           hidden_dims_actor, log_std_init).to(self.device)

        self.actor.action_scale = self.actor.action_scale.to(self.device)
        self.actor.action_bias = self.actor.action_bias.to(self.device)
        self.actor.eval()

    def select_action(self, stacked_state_tensor, evaluate=True): 
        action_env, _, _ = self.actor.sample(stacked_state_tensor, deterministic=evaluate)
        return action_env.detach().cpu().numpy()[0] 

    def load_actor_model(self, actor_path):
        print(f'Loading Actor model from {actor_path}')
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor.eval()
        self.actor.action_scale = self.actor.action_scale.to(self.device)
        self.actor.action_bias = self.actor.action_bias.to(self.device)


def make_test_env(seed):
    """ Creates and seeds a DMC environment for testing. """
    env_name = "humanoid-walk"
    print(f"Creating test environment: {env_name} with seed {seed}")
    env = make_dmc_env(env_name, seed, flatten=True, use_pixels=False)
    env.action_space.seed(seed) 
    return env

class Agent(object):
    def __init__(self, actor_model_path="model.pth", frame_stack_k=1, device_str="cuda"): 
        """
        actor_model_path: Path to the pre-trained actor model.
        frame_stack_k: Number of frames that were stacked during training.
                       This agent will replicate this stacking.
        """
        self.frame_stack_k = frame_stack_k
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        dummy_env = make_test_env(seed=0)
        self.single_frame_obs_shape = dummy_env.observation_space.shape
        self.action_space = dummy_env.action_space 
        dummy_env.close()

        if not isinstance(self.single_frame_obs_shape, tuple) or len(self.single_frame_obs_shape) != 1:
            raise ValueError(f"Expected a 1D observation space for single frames, got {self.single_frame_obs_shape}")
        self.single_frame_dim = self.single_frame_obs_shape[0]
        self.stacked_state_dim = self.single_frame_dim * self.frame_stack_k

        self.agent_internal = TestSACAgent(
            stacked_state_dim=self.stacked_state_dim,
            action_space=self.action_space,
            device=device_str
        )

        self.agent_internal.load_actor_model(actor_model_path)

        self.obs_deque = deque([], maxlen=self.frame_stack_k)
        self.is_initialized = False 

        print(f"Test Agent initialized. Expects single obs, stacks {self.frame_stack_k} frames internally.")
        print(f"Single obs dim: {self.single_frame_dim}, Stacked obs dim for Actor: {self.stacked_state_dim}")


    def _initialize_deque(self, single_observation):
        """Fills the deque with the first observation."""
        single_observation_float = single_observation.astype(np.float32)
        for _ in range(self.frame_stack_k):
            self.obs_deque.append(single_observation_float)
        self.is_initialized = True

    def _get_stacked_observation(self, new_single_observation):
        """Appends new observation and returns the stacked version."""
        if not self.is_initialized:
            self._initialize_deque(new_single_observation)
        else:
            self.obs_deque.append(new_single_observation.astype(np.float32))
        
        return np.concatenate(list(self.obs_deque), axis=0).astype(np.float32)

    def act(self, observation):
        """
        Receives a single observation, stacks it with previous ones,
        and returns an action from the loaded actor model.
        """
        stacked_obs_np = self._get_stacked_observation(observation)
        
        stacked_obs_tensor = torch.FloatTensor(stacked_obs_np).unsqueeze(0).to(self.device)
        
        action = self.agent_internal.select_action(stacked_obs_tensor, evaluate=True)
        return action

    def reset(self): 
        """Resets the internal frame buffer. Call this when the environment resets."""
        self.is_initialized = False
        self.obs_deque.clear()
        print("Test Agent deque reset.")