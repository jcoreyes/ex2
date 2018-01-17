import cv2
from rllab.envs.gym_env import GymEnv, NoVideoSchedule, CappedCubicVideoSchedule, convert_gym_space
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.box import Box
import traceback
import numpy as np
import gym
import gym.wrappers
import gym.envs
import gym.spaces
try:
    import logging
    from gym.wrappers.monitoring import logger as monitor_logger
    monitor_logger.setLevel(logging.WARNING)
except Exception as e:
    traceback.print_exc()
import os
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.misc import logger
import logging
import multiprocessing

class CroppedGymEnv(GymEnv):
    def __init__(self, env_name, register_info=None, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=True, screen_width=84, screen_height=84, frame_skip=1, doom_actionspace='Box',
                 conv=True, client_port=10000, transpose_output=False, stack_frames=False, stack_size=4):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())
        if 'Doom' in env_name:
            import ex2.envs.doom
        if 'Minecraft' in env_name:
            import axe.envs.minecraft

        if register_info:
            try:
                gym.envs.register(**register_info)
            except gym.error.Error:
                traceback.print_exc()

        env = gym.envs.make(env_name)

        if 'Doom' in env_name:
            from ex2.envs.doom.wrappers import SetResolution
            from ex2.envs.doom.wrappers.action_space import ToDiscrete, ToBox
            if doom_actionspace == 'Box':
                wrapper1 = ToBox('minimal')
            else:
                wrapper1 = ToDiscrete('minimal')
            #lock = multiprocessing.Lock()
            #env.configure(lock=lock)
            wrapper2 = SetResolution('160x120')
            env = wrapper2(wrapper1(env))
        if 'Minecraft' in env_name:
            env.init(videoResolution=[screen_width, screen_height], allowContinuousMovement=["move", "turn"],
                     continuous_discrete=False, vision=False,
                     client_pool=[('127.0.0.1', client_port)])

        self.env = env
        self.env_id = env.spec.id
        self.env_name = env_name
        self.frame_skip = frame_skip
        self.stack_frames = stack_frames
        if stack_frames:
            self.channel_size = stack_size
        else:
            self.channel_size = 3

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True


        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        self._log_dir = log_dir
        self._force_reset = force_reset
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.conv = conv
        self.transpose_output = transpose_output
        if conv:
            if self.transpose_output:
                self._observation_space = Box(low=0, high=1, shape=(self.channel_size, screen_width, screen_height))
                #self._observation_space = Box(low=0, high=1, shape=(3* screen_width* screen_height))
            else:
                self._observation_space = Box(low=0, high=1, shape=(screen_width, screen_height, self.channel_size))
        else:
            self._observation_space = Box(low=0, high=1, shape=(self.channel_size,))
        self.last_info = None
        self.last_obs = []


    def reshape(self, obs):
        if self.stack_frames:
            return np.expand_dims(cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (self.screen_width, self.screen_height)), axis=2) / 255
        
        tmp = cv2.resize(obs, (self.screen_width, self.screen_height))
        if self.transpose_output:
            tmp = np.transpose(tmp, [2, 0, 1])

        return tmp / 255
        #return cv2.resize(obs, (self.screen_width, self.screen_height)).transpose(2, 0, 1)

    def step(self, action):
        # if 'Minecraft' in self.env_name:
        #     self.render()
        cum_reward = 0
        for i in range(self.frame_skip):
            next_obs, reward, done, info = self.env.step(action)
            info = {}
            cum_reward += reward
            if done:
                break

        if '_observation' in info:
            if info['_observation'] is None:
                info = self.last_info
            else:
                self.last_info = info
        if self.conv:
            next_obs = self.reshape(next_obs)
            if self.stack_frames:
                self.last_obs.append(next_obs)
                next_obs = np.concatenate(self.last_obs[-self.channel_size:], axis=2)
        else:
            pass
        #print(next_obs)


        return Step(next_obs, cum_reward, done, **info)

    def reset(self):
        if self._force_reset and self.monitoring:
            recorder = self.env._monitor.stats_recorder
            if recorder is not None:
                recorder.done = True
            pass
        if self.conv:
            obs = self.reshape(self.env.reset()) 
            if self.stack_frames:
                self.last_obs = [obs] * self.channel_size
                obs = np.repeat(obs, self.channel_size, axis=2)
            return obs 
        else:
            return self.env.reset()
