import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium
import numpy as np
import cv2
from collections import deque

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

class RenderObservation(gym.ObservationWrapper):
    """直接把 dm_control 渲染結果當作 observation."""
    def __init__(self, env, width=84, height=84, camera_id=0):
        super().__init__(env)
        self.width, self.height = width, height
        self.camera_id = camera_id
        self.env.render_kwargs = {"width": width, "height": height,
                                  "camera_id": camera_id}

        sample = self._process(self.env.render())
        self.observation_space = spaces.Box(
            low=0, high=255, shape=sample.shape, dtype=np.uint8
        )

    def _process(self, frame: np.ndarray) -> np.ndarray:
        # RGB → Gray → Resize → CHW
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height),
                             interpolation=cv2.INTER_AREA)
        chw = resized[None, ...]           # (1, H, W)
        return chw.astype(np.uint8)

    def observation(self, _obs_from_env_dict):
        return self._process(self.env.render())


class FrameStack(gym.Wrapper):
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c * k, h, w), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

def make_dmc_env(
    env_name: str,
    seed: int,
    frame_stack: int = 4,
    flatten: bool = True,
    use_pixels: bool = True,
    width: int = 84,
    height: int = 84,
) -> gym.Env:
    """Create a Gymnasium‑compatible DMC environment with pixel frame stack."""
    domain_name, task_name = env_name.split("-")
    dmc_env = suite.load(domain_name, task_name, task_kwargs={"random": seed})
    env = DmControltoGymnasium(
        dmc_env,
        render_mode="rgb_array",
        render_kwargs={"width": width, "height": height, "camera_id": 0},
    )

    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    if use_pixels:
        env = RenderObservation(env, width=width, height=height)
        if frame_stack > 1:
            env = FrameStack(env, k=frame_stack)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env