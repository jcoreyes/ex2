from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy
import numpy as np

class BatchSampler(BaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            exemplar_cls=None,
            exemplar_args=None,
            bonus_coeff=0,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            sampler_cls=None,
            sampler_args=None,
            eval_first=False,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.current_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        if sampler_cls is None:
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.exemplar = None
        self.exemplar_cls = exemplar_cls
        self.exemplar_args = exemplar_args
        self.bonus_coeff = bonus_coeff
        self.eval_first = eval_first

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def process_paths(self, paths):
        if self.eval_first:
            for path in paths:
                path["raw_rewards"] = np.copy(path["rewards"])
                if self.exemplar is not None:
                    path["bonus_rewards"] = self.exemplar.predict(path)
            if self.exemplar is not None:
                self.exemplar.fit(paths)
        else:
            if self.exemplar is not None:
                self.exemplar.fit(paths)
            for path in paths:
                path["raw_rewards"] = np.copy(path["rewards"])
                if self.exemplar is not None:
                    path["bonus_rewards"] = self.exemplar.predict(path)


        if self.exemplar is not None:
            bonus_rewards = np.concatenate([path["bonus_rewards"].ravel() for path in paths])
            median_bonus = np.median(bonus_rewards)
            mean_discrim = np.mean(1 / (bonus_rewards + 1))
            for path in paths:
                path["bonus_rewards"] -= median_bonus
                path["rewards"] = path["rewards"] + self.bonus_coeff * path["bonus_rewards"]
            logger.record_tabular('Median Bonus', median_bonus)
            logger.record_tabular('Mean Discrim', mean_discrim)

    def train(self):
        if self.exemplar_cls is not None:
            self.exemplar = self.exemplar_cls(**self.exemplar_args)
            self.exemplar.init_rank(0)

        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples(itr)
                self.process_paths(paths)
                samples_data = self.sampler.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        undiscounted_raw_returns = [sum(path["raw_rewards"]) for path in paths]

        num_traj = len(undiscounted_returns)
        sum_return = np.sum(undiscounted_returns)
        min_return = np.min(undiscounted_returns)
        max_return = np.max(undiscounted_returns)
        sum_raw_return = np.sum(undiscounted_raw_returns)
        min_raw_return = np.min(undiscounted_raw_returns)
        max_raw_return = np.max(undiscounted_raw_returns)

        average_return = sum_return / num_traj

        average_raw_return = sum_raw_return / num_traj

        logger.record_tabular('ReturnAverage', average_return)
        logger.record_tabular('ReturnMax', max_return)
        logger.record_tabular('ReturnMin', min_return)
        logger.record_tabular('RawReturnAverage', average_raw_return)
        logger.record_tabular('RawReturnMax', max_raw_return)
        logger.record_tabular('RawReturnMin', min_raw_return)
        if self.exemplar is not None:
            bonuses = np.concatenate([path["bonus_rewards"] for path in paths])
            logger.record_tabular('BonusRewardMax', bonuses.max())
            logger.record_tabular('BonusRewardMin', bonuses.min())
            logger.record_tabular('BonusRewardAverage', bonuses.mean())

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
