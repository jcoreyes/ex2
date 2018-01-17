import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from gym.envs.mujoco import mujoco_env
from gym import utils
from rllab.misc import logger
from rllab.misc.overrides import overrides
from ex2.envs.env_utils import get_asset_xml
#from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv

def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

class SparseHalfCheetahEnv(HalfCheetahEnv, Serializable):
    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        #utils.EzPickle.__init__(self)
        super(SparseHalfCheetahEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        #action = np.clip(action, *self.action_bounds)
        #ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        #run_cost = -1 * self.get_body_comvel("torso")[0]
        #cost = ctrl_cost + run_cost
        #reward = -cost
        done = False

        body_dist = self.get_body_com("torso")[0]
        if abs(body_dist) <= 5.0:
            reward = 0.
        else:
            reward = 1.0
        #import pdb; pdb.set_trace()
        return Step(next_obs, reward, done)
