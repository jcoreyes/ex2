import numpy as np
import multiprocessing as mp
from ex2.parallel_trpo.simple_container import SimpleContainer
from ex2.utils.replay_buffer import SimpleReplayPool

def sample_batch(data, data_size, batch_size):
    idxs = np.random.randint(data_size, size=batch_size)
    return data[idxs]

class Exemplar(object):
    """
    Classify new states vs old states.
    """
    def __init__(
            self,
            state_dim,
            n_action,
            replay_state_dim=None,
            train_itrs=1e3,
            first_train_itrs=5e3,
            batch_size=128,
            replay_size=1e5,
            min_replay_size=1e3,
            log_freq=0.1,
            state_preprocessor=None,
            bonus_form="1/sqrt(p)",
            log_prefix="",
            parallel=False,
            retrieve_sample_size=np.inf,
            decay_within_path=False,
            model_cls=None,
            model_args=None,
            use_actions=False
        ):
        self.state_dim = state_dim
        if state_preprocessor is not None:
            assert state_preprocessor.get_output_dim() == state_dim
            self.state_preprocessor = state_preprocessor
        else:
            self.state_preprocessor = None

        self.n_action = n_action

        # self.hash is for compute_keys()
        # self.hash_list for inc_keys() and query_keys()

        self.bonus_form = bonus_form
        self.log_prefix = log_prefix

        self.parallel = parallel

        self.retrieve_sample_size = retrieve_sample_size
        self.decay_within_path = decay_within_path
        self.unpicklable_list = ["_par_objs","shared_dict", 'replay', 'model']
        self.snapshot_list = [""]

        # logging stats ---------------------------------
        self.rank = None

        self.log_freq = log_freq
        self.train_itrs = int(train_itrs)
        self.batch_size = int(batch_size)
        if use_actions and 'input_dim' in model_args:
            model_args['input_dim'] += n_action
        self.model = model_cls(**model_args)
        self.min_replay_size = min_replay_size
        if replay_state_dim is None:
            replay_state_dim = state_dim
        self.replay = SimpleReplayPool(replay_size, replay_state_dim, n_action, use_actions)
        self.use_actions = use_actions
        self.first_train = True
        self.first_train_itrs = int(first_train_itrs)

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        state = dict()
        for k,v in iter(self.__dict__.items()):
            if k not in self.unpicklable_list:
                state[k] = v
            elif k in self.snapshot_list:
                state[k] = copy.deepcopy(v)
        return state

    # note that self.hash does not need parallelism
    def init_rank(self,rank):
        self.rank = rank

    def init_shared_dict(self, shared_dict):
        self.shared_dict = shared_dict

    def init_par_objs(self,n_parallel):
        n = n_parallel
        shareds = SimpleContainer(
            new_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            total_state_action_count = np.frombuffer(
                mp.RawValue('l'),
                dtype=int,
            )[0],
            max_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            min_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            sum_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            n_steps_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
        )
        barriers = SimpleContainer(
            summarize_count = mp.Barrier(n),
            update_count = mp.Barrier(n),
        )
        self._par_objs = (shareds, barriers)


    def preprocess(self,states):
        if self.state_preprocessor is not None:
            processed_states = self.state_preprocessor.process(states)
        else:
            processed_states = states
        return processed_states


    def fit(self, paths):
        if self.parallel:
            shareds, barriers = self._par_objs
        # Deal with atari multiple frames. Use last frame
        if paths[0]['observations'].shape[1] != self.state_dim:
            obs = np.concatenate([path['observations'][:, -self.state_dim:] for path in paths]).astype(np.float32)
            # Have tested and works
        else:
            obs = np.concatenate([path['observations'] for path in paths]).astype(np.float32)

        actions = np.concatenate([path['actions'] for path in paths]).astype(np.float32)
        if self.use_actions:
            positives = np.concatenate([obs, actions], 1)
        else:
            positives = obs

        if self.replay.size >= self.min_replay_size:
            log_step = self.train_itrs * self.log_freq
            labels = np.expand_dims(np.concatenate([np.ones(self.batch_size), np.zeros(self.batch_size)]), 1).astype(np.float32)

            if self.first_train:
                train_itrs = self.first_train_itrs
                self.first_train = False
            else:
                train_itrs = self.train_itrs

            for train_itr in range(train_itrs):
                pos_batch = sample_batch(positives, positives.shape[0], self.batch_size)
                neg_batch = self.replay.random_batch(self.batch_size)
                x1 = np.concatenate([pos_batch, pos_batch])
                x2 = np.concatenate([pos_batch, neg_batch])
                loss, class_loss, kl_loss = self.model.train_batch(x1, x2, labels)

                #if self.rank == 0 and train_itr % log_step == 0:
                #    print("%.4f %.4f %.4f" %(loss, class_loss, kl_loss))

        self.replay.add_samples(obs, actions)


    def predict(self, path):
        if self.replay.size < self.min_replay_size:
            return np.zeros(len(path['observations']))

        obs = path['observations'].astype(np.float32)
        actions = path['actions'].astype(np.float32)

        # Deal with atari multiple frames
        if obs.shape[1] != self.state_dim:
            obs = obs[:, -self.state_dim:]
        if self.use_actions:
            positives = np.concatenate([obs, actions], 1)
        else:
            positives = obs

        counts = self.model.test(positives)
        # if self.rank == 0:
        #     logger.record_tabular('Average Prob', np.mean(counts))
        #     logger.record_tabular('Average Discrim', np.mean(1/(5.01*counts + 1)))

        if self.bonus_form == "1/n":
            bonuses = 1./counts
        elif self.bonus_form == "1/sqrt(pn)":
            bonuses = 1. / np.sqrt(self.replay.size * counts)
        elif self.bonus_form == "1/sqrt(p)":
            bonuses = 1./np.sqrt(counts)
        elif self.bonus_form == "1/log(n+1)":
            bonuses = 1./np.log(counts + 1)
        elif self.bonus_form == "1/log(n)":
            bonuses = 1. / np.log(counts)
        elif self.bonus_form == "-log(p)":
            bonuses = - np.log(counts)
        else:
            raise NotImplementedError
        return bonuses

    def reset(self):
        self.model.reset_weights()

    def log_diagnostics(self, paths):
        pass
