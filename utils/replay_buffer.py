import numpy as np

class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim, use_actions):
        max_pool_size = int(max_pool_size)
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self.use_actions = use_actions
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
            dtype=np.float32
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
            dtype=np.float32
        )

        self._top = 0
        self._size = 0


    @property
    def observations(self):
        return self._observations

    def add_sample(self, observation, action):
        self._observations[self._top] = observation
        # self._actions[self._top] = action
        self._top = int((self._top + 1) % self._max_pool_size)
        self._size += 1

    def add_samples(self, observations, actions):
        # Observations and actions are each arrays

        n_samples = observations.shape[0]
        if self._top + n_samples >= self._max_pool_size:
            first_size = self._max_pool_size - self._top
            second_size = n_samples - first_size
            self._observations[self._top:] = observations[:first_size]
            self._observations[:second_size] = observations[first_size:]
            if self.use_actions:
                self._actions[self._top:] = actions[:first_size]
                self._actions[:second_size] = actions[first_size:]
        else:
            self._observations[self._top:self._top+n_samples] = observations
            self._actions[self._top:self._top+n_samples] = actions

        self._size += n_samples
        self._top = int((self._top + n_samples) % self._max_pool_size)

    def random_batch(self, batch_size):
        size = min(self._size, self._max_pool_size)
        indices = np.random.randint(0, size, batch_size)
        if self.use_actions:
            return np.concatenate([self._observations[indices], self._actions[indices]], 1)
        else:
            return self._observations[indices]

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self.size

class MulticlassReplayPool(SimpleReplayPool):
    def add_paths(self, paths, class_idx):
        for path in paths:
            obss, acts, rews = path['observations'], path['actions'], path['rewards']
            for t in range(len(obss)-1):
                self.add_sample(obss[t], acts[t], rews[t], class_idx)

class SimpleRecencyReplayPool(SimpleReplayPool):
    def __init__(
            self, recency_threshold=None, recency_func=None, **kwargs):
        super(SimpleRecencyReplayPool, self).__init__(
            **kwargs)
        self._recencies = np.zeros(self._max_pool_size)
        self.recency_count = 0
        self.recency_threshold = recency_threshold

        self.recency_func = None
        if recency_func is not None:
            self.recency_func = getattr(np, recency_func)

    def add_sample(self, observation, action, reward, terminal):
        self._recencies[self._top] = self.recency_count
        super(SimpleRecencyReplayPool, self).add_sample(
            observation, action, reward, terminal)
        self.recency_count += 1

    def add_paths(self, paths):
        for path in paths:
            obss, acts, rews = path['observations'], path['actions'], path['rewards']
            for t in range(len(obss)-1):
                done = int(t == (len(obss)-1))
                self.add_sample(obss[t], acts[t], rews[t], done)

    def random_batch(self, batch_size, recency_threshold=None):
        indices, transition_indices = self.random_batch_indices(batch_size)
        return self.indices_to_batch(indices, transition_indices, recency_threshold=recency_threshold)

    def indices_to_batch(self, indices, transition_indices, recency_threshold=None):
        recencies = self.recency_count - self._recencies[indices]
        if recency_threshold is not None:
            binary_recencies = np.zeros(recencies.shape[0])
            binary_recencies[recencies<recency_threshold] = 1
            recencies = binary_recencies
        if self.recency_func is not None:
            recencies = self.recency_func(recencies)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices],
            recencies=recencies
        )


    def random_batch_balanced(self, batch_size, recent_ratio=0.5, recency_threshold=None):
        if recency_threshold is None:
            recency_threshold = self.recency_threshold
        assert recency_threshold < self._size

        num_recent = round(batch_size * recent_ratio)
        num_old = batch_size - num_recent
        v = BufferView(self._max_pool_size, self._top, self._bottom, self._size)

        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < num_recent:
            index = v.map(np.random.randint(len(v)-recency_threshold, len(v)))
            transition_index = (index+1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1

        count = 0
        while count < num_old:
            index = v.map(np.random.randint(0, len(v)-recency_threshold))
            transition_index = (index+1) % self._max_pool_size
            indices[num_recent + count] = index
            transition_indices[num_recent + count] = transition_index
            count += 1
        
        return self.indices_to_batch(indices, transition_indices, recency_threshold)


class BufferView(object):
    """ Helper class to manage indexing into the replay buffer """
    def __init__(self, capacity, top, bottom, size):
        self.capacity = capacity
        self.top = top
        self.bottom = bottom
        self.size = size

    def map(self, index):
        assert np.all(index < self.size)
        return (index + self.bottom) % self.capacity

    def __len__(self):
        return self.size