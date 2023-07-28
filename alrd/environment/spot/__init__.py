from alrd.environment.spot.spot2d import Spot2DEnv
from alrd.environment.spot.robot_state import SpotState, KinematicState, modify_2d_state
from alrd.environment.spot.mobility_command import MobilityCommand
from alrd.environment.spot.record import Session
from mbse.models.bayesian_dynamics_model import BayesianDynamicsModel
from mbse.utils.model_checkpoint import predict_multistep
from mbse.utils.replay_buffer import Transition, EpisodicReplayBuffer
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R

def predict_states_2d_model(states: List[SpotState], commands: List[MobilityCommand],
                            model: BayesianDynamicsModel, use_history=None, use_action_history=False,
                            include_only=None) -> List[SpotState]:
    obs = np.array([Spot2DEnv.get_obs_from_state(state) for state in states]),
    action = np.array([Spot2DEnv.get_action_from_command(cmd) for cmd in commands]),
    tran = Transition(
        obs=obs[:-1],
        action=action,
        next_obs=obs[1:],
        reward=np.zeros(len(states)-1),
        done=np.zeros(len(states)-1)
    )
    predictions = predict_multistep(tran, model, use_history=use_history, use_action_history=use_action_history, include_only=include_only)
    pred_states = []
    for state, pred in zip(state, predictions):
        pred_states.append(modify_2d_state(state, pred))
    return pred_states

def get_2d_buffer(session: Session) -> EpisodicReplayBuffer:
    buffer = EpisodicReplayBuffer(obs_shape=Spot2DEnv.obs_shape, action_shape=Spot2DEnv.action_shape, max_size=1000000)
    for i in range(len(session)):
        episode = session.get_episode(i)
        x, y, _, qx, qy, qz, qw = episode.obs[0].pose_of_body_in_vision # TODO assumes it starts in the origin, could be different
        angle = R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2]
        obs = np.array([Spot2DEnv.get_obs_from_state_goal(state, np.array([x, y, angle])) for state in episode.obs])
        action = np.array([Spot2DEnv.get_action_from_command(cmd) for cmd in episode.actions])
        tran = Transition(
            obs=obs[:-1],
            action=action,
            next_obs=obs[1:],
            reward=np.array(episode.rewards),
            done=np.array(episode.dones)
        )
        buffer.add(tran)
    return buffer