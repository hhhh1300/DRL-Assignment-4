import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium
import numpy as np

# 20 tasks
DMC_EASY_MEDIUM = [
    "acrobot-swingup",
    "cartpole-balance",
    "cartpole-balance_sparse",
    "cartpole-swingup",
    "cartpole-swingup_sparse",
    "cheetah-run",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "fish-swim",
    "hopper-hop",
    "hopper-stand",
    "pendulum-swingup",
    "quadruped-walk",
    "quadruped-run",
    "reacher-easy",
    "reacher-hard",
    "walker-stand",
    "walker-walk",
    "walker-run",
]

# 8 tasks
DMC_SPARSE = [
    "cartpole-balance_sparse",
    "cartpole-swingup_sparse",
    "ball_in_cup-catch",
    "finger-spin",
    "finger-turn_easy",
    "finger-turn_hard",
    "reacher-easy",
    "reacher-hard",
]

# 7 tasks
DMC_HARD = [
    "humanoid-stand",
    "humanoid-walk",
    "humanoid-run",
    "dog-stand",
    "dog-walk",
    "dog-run",
    "dog-trot",
]

class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.env.render_kwargs={"width": width, "height": height, "camera_id": 0}
        
        tmp = self.env.render()
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=tmp.shape, 
            dtype=np.uint8
        )

    def observation(self, observation):
        pixel_obs = self.env.render()
        return pixel_obs

def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
    use_pixels: bool = True,
) -> gym.Env:
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array", render_kwargs={"width": 256, "height": 256, "camera_id": 0})
    
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    if use_pixels:
        env = PixelObservationWrapper(env)
        
    return env
