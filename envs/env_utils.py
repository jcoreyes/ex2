import os
from collections import namedtuple

import numpy as np

from rllab.envs.gym_env import GymEnv
from rllab.envs.proxy_env import ProxyEnv

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)

class CustomGymEnv(GymEnv):
    def __init__(self, env_name, record_log=False, record_video=False):
        import ex2.envs  # Force register
        ex2.envs.register_custom_envs()
        super(CustomGymEnv, self).__init__(env_name, record_log=record_log, record_video=record_video)

def get_inner_env(env):
    if isinstance(env, ProxyEnv):
        return get_inner_env(env.wrapped_env)
    elif isinstance(env, GymEnv):
        return get_inner_env(env.env)
    return env

def test_env(env, T=100):
    aspace = env.action_space
    env.reset()
    for t in range(T):
        o, r, done, infos = env.step(aspace.sample())
        print('---T=%d---' % t)
        print('rew:', r)
        print('obs:', o)
        env.render()
        if done:
            break
    

if __name__=="__main__":
    from axe.envs.sparse_pendulum import *
    env = SparsePendulum()
    test_env(env)

