# imports -----------------------------------------------------
""" baseline """
from ex2.parallel_trpo.gaussian_conv_baseline import ParallelGaussianConvBaseline
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from ex2.misc.network_args import nips_dqn_args
from ex2.parallel_trpo.conjugate_gradient_optimizer import ParallelConjugateGradientOptimizer
from ex2.parallel_trpo.trpo import ParallelTRPO
from ex2.envs.atari_env import AtariEnv

""" exemplar """
from ex2.models.exemplar import Exemplar
from ex2.models.siamese import SiameseConv

""" others """
from utils.log_utils import get_time_stamp
from ex2.misc.ec2_info import subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy
import lasagne.nonlinearities as NL

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_docker"  # "ec2"
ec2_instance = "c4.8xlarge"
price_multiplier = 3
subnet = "us-west-1b"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:theano" # needs psutils

n_parallel = 1 #2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 50
plot = False
use_gpu = False
sync_s3_pkl = True
sync_s3_log = True
config.USE_TF = False

if "local" in mode and sys.platform == "darwin":
    set_cpu_affinity = False
    cpu_assignments = None
    serial_compile = False
else:
    set_cpu_affinity = True
    cpu_assignments = None
    serial_compile = True

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200]

    @variant
    def frame_skip(self):
        return [4]

    @variant
    def img_size(self):
        return [42]

    @variant
    def game(self):
        return ["venture", "freeway", 'frostbite']

    @variant
    def bonus_coeff(self):
        return [1e-3, 1e-4, 1e-5]

    @variant
    def kl_weight(self):
        return [0.1]

    @variant
    def use_actions(self):
        return [False]


variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params ------------------------------
    # algo
    use_parallel = True
    seed = v["seed"]
    persistent = True
    scale_neg1_1 = False
    img_size = v["img_size"]

    if mode == "local_test":
        batch_size = 500
    elif mode == "local_docker" or mode == 'local':
        batch_size = 1000 #10000
    else:
        if img_size > 50:
            batch_size = 25000
        else:
            batch_size = 100000

    max_path_length = 4500
    discount = 0.99
    n_itr = 500
    step_size = 0.01
    policy_opt_args = dict(
        name="pi_opt",
        cg_iters=10,
        reg_coeff=1e-3,
        subsample_factor=0.1,
        max_backtracks=15,
        backtrack_ratio=0.8,
        accept_violation=False,
        hvp_approach=None,
        num_slices=1, # reduces memory requirement
    )
    network_args = nips_dqn_args
    clip_reward = True  # Clip rewards to be from -1 to 1

    # env
    game = v["game"]
    exp_prefix = "trpo/" + exp_index + "-" + game.split('_')[0]
    env_seed = 1 # deterministic env
    frame_skip = v['frame_skip']
    max_start_nullops = 0
    img_width = img_size
    img_height = img_size
    n_last_screens = 4
    obs_type = "image"
    record_image = False
    record_rgb_image = False
    record_ram = False
    record_internal_state = False

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{game}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        game=game,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    if use_gpu:
        config.USE_GPU = True
        config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"

    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = '0.84'
        n_parallel = 12

        # choose subnet
        config.AWS_NETWORK_INTERFACES = [
            dict(
                SubnetId=subnet_info[subnet]["SubnetID"],
                Groups=subnet_info[subnet]["Groups"],
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ]
    else:
        raise NotImplementedError

    # construct objects ----------------------------------
    game_conversion = dict(venture='Venture-v3', freeway='Freeway-v3', frostbite='Frostbite-v3')
    assert game in game_conversion.keys(), "Invalid game choice"
    gym_game = game_conversion[game]

    env = AtariEnv(
            gym_game,
            force_reset=True,
            seed=env_seed,
            img_width=img_width,
            img_height=img_height,
            n_frames=n_last_screens,
            frame_skip=frame_skip,
            persistent=persistent,
            scale_neg1_1=scale_neg1_1
    )

    policy = CategoricalConvPolicy(
        env_spec=env.spec,
        name="policy",
        **network_args
    )

    # baseline
    network_args_for_vf = copy.deepcopy(network_args)
    network_args_for_vf.pop("output_nonlinearity")
    baseline = ParallelGaussianConvBaseline(
        env_spec=env.spec,
        regressor_args = dict(
            optimizer=ParallelConjugateGradientOptimizer(
                subsample_factor=0.1,
                cg_iters=10,
                name="vf_opt",
            ),
            use_trust_region=True,
            step_size=0.01,
            batchsize=batch_size*10,
            normalize_inputs=True,
            normalize_outputs=True,
            **network_args_for_vf
        )
    )

    ExemplarCls = Exemplar
    ModelCls = SiameseConv
    input_dim = (1, img_width, img_height)
    model_args = dict(input_size=img_width * img_height, img_width=img_width, img_height=img_height,
                      feature_dim=32,
                      hidden_sizes=(64,),
                      kl_weight=v['kl_weight'],
                      use_actions=v['use_actions'],
                      action_size=env.action_space.flat_dim,
                      conv_args=dict(filter_sizes=((4,4), (4,4)),
                                     num_filters=(16,8),
                                     strides=((2,2), (2,2)),
                                     hidden_act=NL.rectify
                                     ),
                      )

    exemplar_args = dict(state_dim=img_width * img_height,
                         n_action=env.action_space.flat_dim,
                         model_cls=ModelCls,
                         model_args=model_args,
                         replay_size=6e4,
                         min_replay_size=1e3,
                         train_itrs=1e3,
                         use_actions=v['use_actions'])
    algo = ParallelTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        discount=discount,
        n_itr=n_itr,
        clip_reward=clip_reward,
        plot=plot,
        optimizer_args=policy_opt_args,
        step_size=step_size,
        set_cpu_affinity=set_cpu_affinity,
        cpu_assignments=cpu_assignments,
        serial_compile=serial_compile,
        n_parallel=n_parallel,
        exemplar_cls=ExemplarCls,
        exemplar_args=exemplar_args,
        bonus_coeff=v["bonus_coeff"]
    )


    if use_parallel:
        run_experiment_lite(
            algo.train(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            seed=seed,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            mode=actual_mode,
            variant=v,
            use_gpu=use_gpu,
            plot=plot,
            sync_s3_pkl=sync_s3_pkl,
            sync_s3_log=sync_s3_log,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
            use_cloudpickle=False,
        )
    else:
        raise NotImplementedError
    if "local" in mode:
        sys.exit(0)

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
