from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from ex2.parallel_trpo.linear_feature_baseline import ZeroBaseline
from ex2.parallel_trpo.trpo import ParallelTRPO
from ex2.misc.ec2_info import instance_info, subnet_info
from rllab import config


from ex2.models.exemplar import Exemplar
from ex2.utils.log_utils import get_time_stamp
from ex2.models.siamese import Siamese

import os

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200, 300, 400]
    @variant
    def env(self):
        return ['SwimmerGather']
    @variant
    def bonus_coeff(self):
        return [1e-4]
    @variant
    def entropy_bonus(self):
        return [0]
    @variant
    def train_itrs(self):
        return [1e3]
    @variant
    def n_parallel(self):
        return [10]
    @variant
    def reset_freq(self):
        return [0]
    @variant
    def hidden_sizes(self):
        return [(16,)]
    @variant
    def exemplar_learning_rate(self):
        return [1e-4]
    @variant
    def bonus_form(self):
        return ["-log(p)"]
    @variant
    def use_actions(self):
        return [False]
    @variant
    def kl_weight(self):
        return [10]
    @variant
    def eval_first(self):
        return [False]


# mode = "local_docker"
ec2_instance = "c4.8xlarge"
price_multiplier = 1
subnet = "us-west-1b"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:theano"  # needs psutils

actual_mode = "local"
# configure instance
info = instance_info[ec2_instance]
config.AWS_INSTANCE_TYPE = ec2_instance
config.AWS_SPOT_PRICE = '0.84'


n_parallel = 1

# choose subnet
config.AWS_NETWORK_INTERFACES = [
    dict(
        SubnetId=subnet_info[subnet]["SubnetID"],
        Groups=subnet_info[subnet]["Groups"],
        DeviceIndex=0,
        AssociatePublicIpAddress=True,
    )
]

variants = VG().variants()

for v in variants:
    exp_index = os.path.basename(__file__).split('.')[0]
    exp_prefix = "trpo/" + exp_index + "-" + v["env"]

    exp_name = "{exp_index}_{time}_{env}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env=v["env"],
    )


    env = normalize(SwimmerGatherEnv())
    PolicyCls = GaussianMLPPolicy
    policy_args = dict(
        env_spec=env.spec,
        init_std=2.0,
        hidden_sizes=(64, 32)
    )
    max_path_length = 500
    sampler_args = dict(buffer_oversize=1.2)  # (max batch_size if whole_paths=True)

    baseline = ZeroBaseline(env_spec=env.spec)

    model_args = dict(input_dim=env.observation_space.flat_dim,
                      feature_dim=v['hidden_sizes'][-1]//2,
                      hidden_sizes=v['hidden_sizes'],
                      l2_reg=0,
                      learning_rate=v['exemplar_learning_rate'],
                      kl_weight=v['kl_weight'],
                      env_name='Cheetah'
                      )
    exemplar_args = dict(state_dim=env.observation_space.flat_dim, #1,
                         replay_state_dim=env.observation_space.flat_dim,
                         n_action=env.action_space.flat_dim,
                         model_cls=Siamese,
                         model_args=model_args,
                         replay_size=50000*10,
                         min_replay_size=1e4,
                         train_itrs=v["train_itrs"],
                         bonus_form=v["bonus_form"],
                         use_actions=v["use_actions"])

    algo = ParallelTRPO(
        env=env,
        policy=PolicyCls(**policy_args),
        baseline=baseline,
        batch_size=50000,
        max_path_length=max_path_length,
        #whole_paths=False,
        n_parallel=v['n_parallel'],
        n_itr=2000,
        discount=0.99,
        step_size=0.01,
        exemplar_cls=Exemplar,
        exemplar_args=exemplar_args,
        bonus_coeff=v['bonus_coeff'],
        entropy_bonus=v['entropy_bonus'],
        reset_freq=v['reset_freq'],
        eval_first=v['eval_first'],
        plot_exemplar=False,
        set_cpu_affinity=True,
        cpu_assignments=None,
        serial_compile=True
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"], 
        variant=v,
        mode=actual_mode,
        sync_s3_log=True,
        use_cloudpickle=False,
        sync_log_on_termination=True,
        sync_all_data_node_to_s3=True,
        snapshot_mode='gap',
        snapshot_gap=500

    )
    if 'local' in actual_mode:
        break
