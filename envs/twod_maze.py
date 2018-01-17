import numpy as np
from gym import utils

from ex2.envs.env_utils import get_asset_xml
from ex2.envs.twod_mjc_env import TwoDEnv

INIT_POS = np.array([0.15,0.15])
TARGET = np.array([0.15, -0.15])
DIST_THRESH = 0.12

class TwoDMaze(TwoDEnv, utils.EzPickle):
    def __init__(self, verbose=False):
        self.verbose = verbose
        utils.EzPickle.__init__(self)
        TwoDEnv.__init__(self, get_asset_xml('twod_maze.xml'), 2, xbounds=[-0.3,0.3], ybounds=[-0.3,0.3])

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        if pos[0]>0.0 and pos[1]<-.08:
            dist = np.sum(np.abs(pos-TARGET))
            reward = - (dist) * 5
        else:
            reward = -1

        if self.verbose:
            print(pos, reward)
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos]).ravel() - INIT_POS

    def viewer_setup(self):
        v = self.viewer

