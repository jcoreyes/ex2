from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(id='Ex2OneDPoint-v0', entry_point='ex2.envs.oned_point:OneDPoint')
    register(id='Ex2TwoDPoint-v0', entry_point='ex2.envs.twod_point:TwoDPoint')
    register(id='Ex2TwoDMaze-v0', entry_point='ex2.envs.twod_maze:TwoDMaze')
    register(id='Ex2TwoDCorr-v0', entry_point='ex2.envs.twod_corridor:TwoDCorridor')
    register(id='Ex2SparsePendulum-v0', entry_point='ex2.envs.sparse_pendulum:SparsePendulum')
    register(id='Ex2SparseCheetah-v0', entry_point='ex2.envs.half_cheetah_env:SparseHalfCheetahEnv')
    LOGGER.info("Finished registering custom gym environments")

