"""Wrapper around OpenAI Gym Atari environments. Similar to rllab.envs.gym_env.
   Pre-processes raw Atari frames (210x160x3) into 84x84x4 grayscale images
   across the last four timesteps."""

import gym, gym.envs, gym.spaces
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from sandbox.sandy.envs.core_env import CoreEnv
from sandbox.sandy.envs.gym_env import GymEnv
from sandbox.sandy.misc.util import suppress_stdouterr
from sandbox.sandy.shared.ale_compatibility import set_gym_seed

from sandbox.haoran.ale_python_interface import ALEInterface
from sandbox.sandy.envs.default_params import DEFAULT_IMG_HEIGHT, DEFAULT_IMG_WIDTH, \
    DEFAULT_N_FRAMES, DEFAULT_FRAMESKIP, DEFAULT_PERSISTENT, DEFAULT_SCALE_NEG1_1

DEFAULT_USE_UPDATED_ALE = True


def get_base_env(obj):
    # Find level of obj that contains base environment, i.e., the env that links to ALE
    # (New version of Monitor in OpenAI gym adds an extra level of wrapping)
    while True:
        if not hasattr(obj, 'env'):
            return None
        if hasattr(obj.env, 'ale'):
            return obj.env
        else:
            obj = obj.env


class AtariEnv(GymEnv, CoreEnv):
    def __init__(self, env_name, record_video=False, video_schedule=None, \
                 log_dir=None, record_log=True, force_reset=False, \
                 **kwargs):
        # persistent = True if the adversarial changes should stay in the history
        #                   of N_FRAMES
        # A persistent adversary attacks before the observation is stored in history
        # A non-persistent adversary attacks after observation is stored in history.
        # before it is fed through the policy neural net (this seems contrived)

        GymEnv.__init__(self, env_name, record_video=record_video, \
                        video_schedule=video_schedule, log_dir=log_dir, \
                        record_log=record_log, force_reset=force_reset, **kwargs)
        CoreEnv.__init__(self, **kwargs)  # Necessary to call this -- sets up attributes

        self.frame_dropout = kwargs.get("frame_dropout", 0)

        frameskip = kwargs.get('frame_skip', DEFAULT_FRAMESKIP)
        self.env.env.env.frameskip = frameskip

        self.base_env = get_base_env(self.env)

        self.use_updated_ale = kwargs.get("use_updated_ale", DEFAULT_USE_UPDATED_ALE)
        if self.use_updated_ale:
            # Replace GymEnv's ALE, since it doesn't save and load states properly
            with suppress_stdouterr():
                self.base_env.ale = ALEInterface()
            self.base_env.ale.setFloat(b'repeat_action_probability', 0.0)

        # Set seed, now that correct ALE has been loaded (if use_updated_ale = True)
        seed = None
        if 'seed' in kwargs:
            seed = kwargs['seed']
        set_gym_seed(self.base_env, seed=seed)

        if self.use_updated_ale:
            # Adjust size of buffer (for ale.getScreenRGB(_buffer)),
            # to be compatible with newer ALE interface
            screen_width, screen_height = self.base_env.ale.getScreenDims()
            self.base_env._buffer = np.empty((screen_height, screen_width, 3), dtype=np.uint8)

        self.reset_internals()

    @overrides
    def step(self, action):
        # next_obs should be Numpy array of shape (210,160,3)
        next_obs, reward, done, info = self.env.step(action)
        if self.use_updated_ale:
            next_obs = next_obs[:, :, [2, 1, 0]]

        if self.frame_dropout > 0:
            if np.random.rand(1)[0] < self.frame_dropout:
                # zero-out observation
                next_obs = np.zeros(next_obs.shape).astype(np.uint8)

        self.update_internals(next_obs, action, reward, done)

        # Sanity check: visualize self.observation, to make sure it's possible
        # for human to play using this simplified input
        # import cv2
        # vis_obs = (self.observation + 1.0) / 2.0  # scale to be from [0,1] for visualization
        # for i in range(vis_obs.shape[0]):
        #    cv2.imshow(str(i), vis_obs[i,:,:])
        # cv2.waitKey()

        return Step(self.observation, reward, done, **info)

    @overrides
    def reset(self):
        obs = GymEnv.reset(self)  # self.env.reset()
        self.reset_internals(obs=obs)
        return self.observation

    def save_state(self):
        self.snapshot = {}
        self.snapshot['ale'] = self.base_env.ale.cloneSystemState()
        self.snapshot['done'] = self.env._monitor.stats_recorder.done
        CoreEnv.save_state(self)

    def restore_state(self):
        assert self.snapshot is not None
        for k, v in self.snapshot.items():
            if k == 'ale':
                self.base_env.ale.restoreSystemState(v)
            setattr(self, k, v)
        self.env._monitor.stats_recorder.done = self.snapshot['done']
        self.snapshot = None

    def equiv_to(self, other_env):
        if not CoreEnv.equiv_to(self, other_env):
            return False
        if self.base_env.ale.getFrameNumber() != other_env.base_env.ale.getFrameNumber():
            print("Frame numbers not equal")
            return False
        if self.base_env.ale.getInt(b'random_seed') != other_env.base_env.ale.getInt(b'random_seed'):
            print("Different random seed")
            return False
        if self.base_env.ale.getFloat(b'repeat_action_probability') != other_env.base_env.ale.getFloat(
                b'repeat_action_probability'):
            print("Different repeat action prob")
            return False

        return True
