from datetime import datetime
from scipy.spatial.transform import Rotation
from mbse.utils.replay_buffer import ReplayBuffer, Transition
import time
import numpy as np

def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def get_transition_from_buffer(buffer: ReplayBuffer):
    t = Transition(
        buffer.obs[:buffer.size],
        buffer.action[:buffer.size],
        buffer.next_obs[:buffer.size],
        buffer.reward[:buffer.size],
        buffer.done[:buffer.size]
    )
    return t

def rotate_2d_vector(vector, angle):
    """
    Rotates a 2d vector by angle degrees
    :param vector: 2d vector
    :param angle: angle in degrees
    :return: rotated vector
    """
    return Rotation.from_euler('z', angle, degrees=True).as_matrix()[:2, :2] @ vector

def sleep_ms(miliseconds):
    start = time.time()
    while time.time() - start < miliseconds / 1000:
        time.sleep(0.001)

def convert_to_cos_sin(transition: Transition):
    """
    Converts a transitions observations from (x, y, theta, vx, vy, va) to (x, y, cos(theta), sin(theta), vx, vy, va)
    :param transition: transition to convert
    :return: converted transition
    """
    obs = np.zeros((transition.obs.shape[0], 7))
    obs[:,:2] = transition.obs[:,:2]
    obs[:,2] = np.cos(transition.obs[:,2] * np.pi / 180)
    obs[:,3] = np.sin(transition.obs[:,2] * np.pi / 180)
    obs[:,4:] = transition.obs[:,3:]
    next_obs = np.zeros((transition.next_obs.shape[0], 7))
    next_obs[:,:2] = transition.next_obs[:,:2]
    next_obs[:,2] = np.cos(transition.next_obs[:,2] * np.pi / 180)
    next_obs[:,3] = np.sin(transition.next_obs[:,2] * np.pi / 180)
    next_obs[:,4:] = transition.next_obs[:,3:]
    return Transition(obs, transition.action, next_obs, transition.reward, transition.done)

def convert_buffer_to_cos_sin(buffer: ReplayBuffer):
    """
    Takes a buffer with observations (x, y, theta, vx, vy, va) and returns a new one with observations (x, y, cos(theta), sin(theta), vx, vy, va)
    :param transition: transition to convert
    :return: converted transition
    """
    assert buffer.obs_shape == (6,)
    new_buffer = ReplayBuffer(
        obs_shape=(7,),
        action_shape=buffer.action_shape,
        max_size=buffer.max_size,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas
    )
    t = get_transition_from_buffer(buffer)
    t = convert_to_cos_sin(t)
    new_buffer.add(t)
    return new_buffer