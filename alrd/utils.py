from datetime import datetime
from scipy.spatial.transform import Rotation
from mbse.utils.replay_buffer import ReplayBuffer, Transition
import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Sequence

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

def convert_to_cos_sin(obs: np.ndarray):
    """
    Converts observations from (x, y, theta, vx, vy, va) to (x, y, cos(theta), sin(theta), vx, vy, va)
    :param obs: observation to convert
    :return: converted observation
    """
    newobs = np.zeros((*obs.shape[:-1], 7))
    newobs[...,[0,1,4,5,6]] = obs[...,[0,1,3,4,5]]
    newobs[...,2] = np.cos(obs[...,2] * np.pi / 180)
    newobs[...,3] = np.sin(obs[...,2] * np.pi / 180)
    return newobs

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
    transition = get_transition_from_buffer(buffer)
    new_transition = Transition(
        convert_to_cos_sin(transition.obs),
        transition.action,
        convert_to_cos_sin(transition.next_obs),
        transition.reward,
        transition.done)
    new_buffer.add(new_transition)
    return new_buffer

def convert_buffer_to_global(buffer: ReplayBuffer):
    """
    Takes a buffer with actions in local reference frame and returns a new one with actions in global reference frame
    :param buffer: buffer to convert
    :return: converted buffer
    """
    assert buffer.obs_shape == (7,)
    new_buffer = ReplayBuffer(
        obs_shape=buffer.obs_shape,
        action_shape=buffer.action_shape,
        max_size=buffer.max_size,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas
    )
    transition = get_transition_from_buffer(buffer)
    rotation = np.array([[transition.obs[:,2], -transition.obs[:,3]], [transition.obs[:,3], transition.obs[:,2]]]).transpose((2,0,1))
    new_action = np.array(transition.action)
    new_action[...,:2] = (rotation @ transition.action[...,:2,None]).squeeze(-1)
    new_transition = Transition(
        transition.obs,
        new_action,
        transition.next_obs,
        transition.reward,
        transition.done)
    new_buffer.add(new_transition)
    return new_buffer

def moving_average_filter(x, window_size: int):
    x = np.stack((np.zeros((window_size, *x.shape[1:])), x))
    cumsum = np.cumsum(x, dtype=float, axis=0)
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
def moving_average_filter_buffer(buffer: ReplayBuffer, window_size: int, episode_len: int, action_window_size: int = 1, obs_idx: Optional[Sequence[int]]=None):
    if obs_idx is None:
        obs_idx = range(t.obs.shape[1])
    def apply_filter(data, window_size):
        obs = np.vstack((np.zeros((window_size, *data.shape[1:])), data))
        cumsum = np.cumsum(obs, dtype=float, axis=0)
        obs = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        return obs

    new_buffer = ReplayBuffer(
        obs_shape=buffer.obs_shape,
        action_shape=buffer.action_shape,
        max_size=buffer.max_size,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas
    )
    t = get_transition_from_buffer(buffer)
    for i in range(0, buffer.size, episode_len):
        start = i
        end = i + episode_len
        if window_size > 1:
            obs = np.vstack((t.obs[start:end], t.next_obs[end-1]))
            obs[:,obs_idx] = apply_filter(obs[:,obs_idx], window_size)
            new_obs = obs[:-1]
            new_next_obs = obs[1:]
        else:
            new_obs = t.obs[start:end]
            new_next_obs = t.next_obs[start:end]
        if action_window_size > 1:
            new_action = apply_filter(t.action[start:end], action_window_size)
        else:
            new_action = t.action[start:end]
        new_t = Transition(new_obs, new_action, new_next_obs, t.reward[start:end], t.done[start:end])
        new_buffer.add(new_t)
    return new_buffer

def strided_median_filter_buffer(buffer: ReplayBuffer, window_size: int, episode_len: int):
    # assumes episode_len is a multiple of window_size
    new_buffer = ReplayBuffer(
        obs_shape=buffer.obs_shape,
        action_shape=buffer.action_shape,
        max_size=buffer.max_size,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas
    )
    t = get_transition_from_buffer(buffer)
    for i in range(0, buffer.size, episode_len):
        start = i
        end = i + episode_len
        obs = np.array(t.obs[start:end]).reshape(-1, window_size, t.obs.shape[-1])
        obs = np.median(obs, axis=-2)
        new_obs = obs[:-1]
        new_next_obs = obs[1:]
        new_t = Transition(
            new_obs,
            t.action[start:end-window_size:window_size],
            new_next_obs,
            t.reward[start+window_size:end:window_size],
            t.done[start+2*window_size-1:end:window_size])
        new_buffer.add(new_t)
    return new_buffer

# borrowed from: https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
def non_uniform_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length
        as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed

def uniform_savgol(y, window_size, degree):
    new_y = np.zeros_like(y)
    for i in range(y.shape[-1]):
        new_y[:, i] = non_uniform_savgol(np.arange(len(y)), y[:, i], window_size, degree)
    return new_y

def slice_buffer(buffer: ReplayBuffer, start: int, end: int):
    new_buffer = ReplayBuffer(
        obs_shape=buffer.obs_shape,
        action_shape=buffer.action_shape,
        max_size=buffer.max_size,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas
    )
    t = get_transition_from_buffer(buffer)
    new_t = Transition(
        t.obs[start:end],
        t.action[start:end],
        t.next_obs[start:end],
        t.reward[start:end],
        t.done[start:end]
    )
    new_buffer.add(new_t)
    return new_buffer

def get_dims(buffer: ReplayBuffer, idx):
    new_buffer = ReplayBuffer(
        obs_shape=(*buffer.obs_shape[:-1], len(idx)),
        action_shape=buffer.action_shape,
        max_size=buffer.max_size,
        normalize=buffer.normalize,
        action_normalize=buffer.action_normalize,
        learn_deltas=buffer.learn_deltas
    )
    t = get_transition_from_buffer(buffer)
    new_t = Transition(
        t.obs[...,idx],
        t.action,
        t.next_obs[...,idx],
        t.reward,
        t.done
    )
    new_buffer.add(new_t)
    return new_buffer

class FlipX:
    """
    Flip data along X axis (front-rear axis)
    WARNING: should only be used for model learning, since the rewards will not correspond to the new observations
    """
    def __init__(self, buffer: ReplayBuffer, flip_y) -> None:
        self.buffer = buffer
        self.flip_y = jnp.array([0, flip_y, 0, 0, 0, 0, 0])
    
    def sample(self, rng, batch_size: int = 256):
        samplerng, rng = jax.random.split(rng)
        tran = self.buffer.sample(samplerng, batch_size)
        idx = jax.random.randint(rng, (batch_size, 1), 0, 2)
        obs = jnp.where(idx, tran.obs, (tran.obs - self.flip_y) * jnp.array([1, -1, 1, -1, 1, -1, -1]) + self.flip_y)
        action = jnp.where(idx, tran.action, tran.action * jnp.array([1, -1, -1]))
        next_obs = jnp.where(idx, tran.next_obs, (tran.next_obs - self.flip_y) * jnp.array([1, -1, 1, -1, 1, -1, -1]) + self.flip_y)
        return Transition(obs, action, next_obs, tran.reward, tran.done)
