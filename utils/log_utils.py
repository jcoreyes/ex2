import os
import errno
import contextlib
import matplotlib.image

import rllab.misc.logger as logger
import matplotlib.pyplot as plt
import numpy as np

from ex2.envs.env_utils import get_inner_env

IMG_DIR = 'img'
VIDEO_DIR = 'video'
_snapshot_dir = None

def get_time_stamp():
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
    return timestamp

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_snapshot_dir(dirname):
    global _snapshot_dir
    _snapshot_dir = dirname


def get_snapshot_dir():
    return logger.get_snapshot_dir() or _snapshot_dir or None


def logger_active():
    return get_snapshot_dir() is not None


def get_img_dir():
    if not logger_active():
        raise NotImplementedError()
    dirname = os.path.join(get_snapshot_dir(), IMG_DIR)
    mkdir_p(dirname)
    return dirname

def get_video_dir():
    if not logger_active():
        raise NotImplementedError()
    dirname = os.path.join(get_snapshot_dir(), VIDEO_DIR)
    mkdir_p(dirname)
    return dirname


def record_image(name, img, itr=None, cmap='afmhot'):
    if not logger_active():
        return
    if itr is not None:
        name += '_itr%d' % itr
    filename = os.path.join(get_img_dir(), name)
    matplotlib.image.imsave(filename+'.png', img, cmap=cmap)

def record_fig(name, itr=None):
    if not logger_active():
        return
    if itr is not None:
        name += '_itr%d' % itr
    filename = os.path.join(get_img_dir(), name)
    plt.savefig(filename)

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def plot_rollout(name, env, agent, max_path_length=np.inf, itr=None, fps=30):
    import imageio
    if itr is not None:
        name += '_itr%d' % itr
    fname = os.path.join(get_video_dir(), name)+'.gif'

    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0

    with imageio.get_writer(fname, fps=fps) as writer:
        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            observations.append(env.observation_space.flatten(o))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

            env.render()
            data, width, height = get_inner_env(env).get_viewer().get_image()
            data = np.fromstring(data, dtype=np.uint8)
            rgb = np.array([np.flipud(data[i::3].reshape(width, height)) for i in range(3)])

            rgb = np.transpose(rgb, [1, 2, 0])
            writer.append_data(np.array(rgb))
    return
