import matplotlib.pyplot as plt
import numpy as np
import os
from ex2.utils import log_utils

MARKERS = ['o', 's', '8', 'D', '4']

def plot_2dtrajectory(states, score=None, marker='o', cmap='afmhot'):
    N, d = states.shape
    assert d==2

    if score is not None:
        if isinstance(score, np.ndarray):
            c = score
        else:  # assume function...
            c = np.array([score(states[i]) for i in range(N)])
    else:
        c = 1

    x = states[:,0]
    y = -states[:,1]
    plt.scatter(x, y, 10., c=c, edgecolors='none', marker=marker, cmap=cmap)

def plot_1dtrajectory(states, score=None, marker='o', cmap='afmhot'):
    from scipy.stats import binned_statistic
    N, d = states.shape

    if score is not None:
        if isinstance(score, np.ndarray):
            c = score
        else:  # assume function...
            c = np.array([score(states[i]) for i in range(N)])
    else:
        c = 1

    x = states[:,-3]
    stats, bin_edges, binnumber = binned_statistic(x, c, statistic='mean', bins=30)
    #import pdb; pdb.set_trace()
    #y = -states[:,1]
    bin_edges += (bin_edges[1] - bin_edges[0]) / 2
    plt.scatter(bin_edges[:-1], stats)

def plot_paths(paths, score_func=None, cmap='afmhot'):
    all_states = []
    all_scores = []
    for i, path in enumerate(paths):
        marker = MARKERS[i % len(MARKERS)]
        states = path['observations']
        scores = path['bonus_rewards'] if 'bonus_rewards' in path else score_func
        all_states.append(states)
        all_scores.append(scores)
        #plot_trajectory(states, score=scores, cmap=cmap, marker=marker)
    all_states = np.concatenate(all_states)
    all_scores = np.concatenate(all_scores)
    if all_states.shape[1] == 2:
        plot_2dtrajectory(all_states, score=all_scores, cmap=cmap, marker='o')
    else:
        plot_1dtrajectory(all_states, score=all_scores, cmap=cmap, marker='o')


def log_paths(paths, fname, itr=None):
    fig = plt.figure()
    fig.add_subplot(1,1,1, axisbg='#AAAAAA')
    plot_paths(paths)
    log_utils.record_fig(fname, itr=itr)


def generate_test_traj(x0=np.array([0,0]), T=50, var=1e-3):
    states = [x0]
    x = x0
    old_n = np.zeros_like(x)
    for _ in range(T):
        new_n = np.random.randn(2) * var
        n = 0.9*old_n + new_n
        x = x + n
        states.append(x)
        old_n = n
    return np.array(states)

def generate_paths(n=5, T=50):
    paths = []
    for _ in range(n):
        paths.append({
            'observations': generate_test_traj(T=T),
            'score': np.linspace(0,1,num=T)
        })
    return paths


if __name__ == "__main__":
    paths = generate_paths()
    score_func = lambda x: 10*np.linalg.norm(x)
    plt.show()
