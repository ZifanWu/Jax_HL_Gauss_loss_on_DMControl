# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

import copy
from typing import Dict, Optional, OrderedDict

import dm_env
import numpy as np
import gym
from dm_control import suite
from gym import core, spaces

from jaxrl.wrappers.common import TimeStep


# def dmc_spec2gym_space(spec):
#     if isinstance(spec, OrderedDict) or isinstance(spec, dict):
#         spec = copy.copy(spec)
#         for k, v in spec.items():
#             spec[k] = dmc_spec2gym_space(v)
#         return spaces.Dict(spec)
#     elif isinstance(spec, dm_env.specs.BoundedArray):
#         return spaces.Box(low=spec.minimum,
#                           high=spec.maximum,
#                           shape=spec.shape,
#                           dtype=spec.dtype)
#     elif isinstance(spec, dm_env.specs.Array):
#         return spaces.Box(low=-float('inf'),
#                           high=float('inf'),
#                           shape=spec.shape,
#                           dtype=spec.dtype)
#     else:
#         raise NotImplementedError

def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        spec = copy.copy(spec)
        for k, v in spec.items():
            spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low,
                          high=high,
                          shape=spec.shape,
                          dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float('-inf')
            high = float('inf')
        else:
            raise ValueError()

        return spaces.Box(low=low,
                          high=high,
                          shape=spec.shape,
                          dtype=spec.dtype)
    else:
        raise NotImplementedError


class DMCEnv(core.Env):

    def __init__(self,
                 domain_name: Optional[str] = None,
                 task_name: Optional[str] = None,
                 env: Optional[dm_env.Environment] = None,
                 task_kwargs: Optional[Dict] = {},
                 environment_kwargs=None):
        assert 'random' in task_kwargs, 'Please specify a seed, for deterministic behaviour.'
        assert (
            env is not None
            or (domain_name is not None and task_name is not None)
        ), 'You must provide either an environment or domain and task names.'

        if env is None:
            if 'duplo' in domain_name:
                from dm_control import manipulation
                env = manipulation.load(task_name + "_vision")
            else:
                env = suite.load(domain_name=domain_name,
                                task_name=task_name,
                                task_kwargs=task_kwargs,
                                environment_kwargs=environment_kwargs)

        self._env = env
        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(
            self._env.observation_spec())
        # self._size = (64, 64)
        # self._ignored_keys = []
        # for key, value in self._env.observation_spec().items():
        #     if value.shape == (0,):
        #         print(f"Ignoring empty observation key '{key}'.")
        #         self._ignored_keys.append(key)

        self.seed(seed=task_kwargs['random'])

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    # @property
    # def observation_space(self):
    #     spaces = {
    #         "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
    #         "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
    #         "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
    #         "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
    #         "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
    #     }
    #     for key, value in self._env.observation_spec().items():
    #         if key in self._ignored_keys:
    #             continue
    #         if value.dtype == np.float64:
    #             spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
    #         elif value.dtype == np.uint8:
    #             spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
    #         else:
    #             raise NotImplementedError(value.dtype)
    #     return spaces

    def step(self, action: np.ndarray) -> TimeStep:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0
        done = time_step.last()
        obs = time_step.observation

        info = {}
        if done and time_step.discount == 1.0:
            info['TimeLimit.truncated'] = True

        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        return time_step.observation

    def render(self,
               mode='rgb_array',
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        return self._env.physics.render(height=height,
                                        width=width,
                                        camera_id=camera_id)
