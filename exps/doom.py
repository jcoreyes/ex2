from ex2.algos.trpo import TRPO
from rllab.envs.normalized_env import normalize

from rllab.misc.instrument import stub, run_experiment_lite

from rllab.policies.categorical_conv_policy import CategoricalConvPolicy


import lasagne.nonlinearities as NL
from ex2.envs.cropped_gym_env import CroppedGymEnv
from ex2.models.exemplar import Exemplar
from ex2.models.siamese import SiameseConv
from ex2.utils.log_utils import get_time_stamp
from ex2.parallel_trpo.linear_feature_baseline import ZeroBaseline
import os

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200, 300, 400]
    @variant
    def env(self):
        return ['doommaze']
    @variant
    def bonus_coeff(self):
        return [2e-4]
    @variant
    def entropy_bonus(self):
        return [1e-5]
    @variant
    def train_itrs(self):
        return [1000]
    @variant
    def n_parallel(self):
        return [4]
    @variant
    def reset_freq(self):
        return [0]
    @variant
    def hidden_sizes(self):
        return [(32,)]
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
        return [0.001]
    @variant
    def eval_first(self):
        return [False]


variants = VG().variants()

for v in variants:
    exp_index = os.path.basename(__file__).split('.')[0]
    exp_prefix = "trpo/" + exp_index + "-" + v["env"]

    exp_name = "{exp_index}_{time}_{env}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env=v["env"],
    )

    img_width = 32
    img_height = 32
    env = CroppedGymEnv(env_name='ex2/DoomMyWayHomeCustom-v0',
                        screen_height=img_height, screen_width=img_width ,
                        record_video=True, frame_skip=10, conv=True,
                        transpose_output=True,
                        doom_actionspace=None)
    env = normalize(env)

    network_args = dict(
        conv_filters=[16,16],
        conv_filter_sizes=[4,4],
        conv_strides=[4,4],
        conv_pads=[(0,0)]*2,
        hidden_sizes=[32],
        hidden_nonlinearity=NL.rectify,
        output_nonlinearity=NL.softmax,
    )
    policy = CategoricalConvPolicy(
        env_spec=env.spec,
        name="policy",
        **network_args
    )
    batch_size = 4000
    max_path_length = batch_size

    baseline = ZeroBaseline(env_spec=env.spec)
    channel_size = 3
    model_args = dict(input_size=channel_size*img_width * img_height, img_width=img_width, img_height=img_height,
                      channel_size=channel_size,
                      feature_dim=v['hidden_sizes'][-1]//2,
                      hidden_sizes=v['hidden_sizes'],
                      l2_reg=0,
                      learning_rate=v['exemplar_learning_rate'],
                      hidden_act=NL.tanh,
                      kl_weight=v['kl_weight'],
                      set_norm_constant=1,
                      conv_args=dict(filter_sizes=((4, 4), (4, 4)),
                                     num_filters=(16, 16),
                                     strides=((2, 2), (2, 2)),
                                     hidden_act=NL.tanh) # TODO Try Relu
                      )
    exemplar_args = dict(state_dim=env.observation_space.flat_dim, #1,
                         replay_state_dim=env.observation_space.flat_dim,
                         n_action=env.action_space.flat_dim,
                         model_cls=SiameseConv,
                         model_args=model_args,
                         replay_size=5000*50,
                         min_replay_size=4000*2,
                         train_itrs=v["train_itrs"],
                         first_train_itrs=2000,
                         bonus_form=v["bonus_form"],
                         use_actions=v["use_actions"],
                         )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        #whole_paths=False,
        n_parallel=v['n_parallel'],
        n_itr=300,
        discount=0.99,
        step_size=0.01,
        exemplar_cls=Exemplar,
        exemplar_args=exemplar_args,
        bonus_coeff=v['bonus_coeff'],
        entropy_bonus=v['entropy_bonus'],
        eval_first=v['eval_first']
        #sampler_cls=BatchSampler,
        #sampler_arg=sampler_args,
    )
    run_experiment_lite(
        algo.train(),
        #use_gpu=True,
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"], 
        variant=v,
        mode='local',
        sync_s3_log=True,
        use_cloudpickle=False,
        sync_log_on_termination=True,
        sync_all_data_node_to_s3=True,
        snapshot_mode='gap',
        snapshot_gap=50

    )
