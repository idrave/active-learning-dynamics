from typing import Any, List
import jax
import jax.numpy as jnp
from alrd.utils.utils import rotate_2d_vector_jax
from mbse.utils.replay_buffer import EpisodicReplayBuffer
from abc import ABC, abstractmethod
from flax import struct
import functools

def rotate_obs(center, obs, angle_change):
    pos0 = obs[:2] - center
    angle = jnp.arctan2(obs[3], obs[2])
    newpos = rotate_2d_vector_jax(pos0, angle_change, degrees=False) + center
    newangle = jnp.expand_dims(angle + angle_change, -1)
    newvel = rotate_2d_vector_jax(obs[4:6], angle_change, degrees=False)
    w = jnp.expand_dims(obs[6], -1)
    return jnp.concatenate([newpos, jnp.cos(newangle), jnp.sin(newangle), newvel, w])

class Rotation:
    """
    Rotate the observations by a random angle
    Does not modify done
    Assumes position state velocity is in global frame and actions in local frame
    """
    def __init__(self, rng, reward_model) -> None:
        self.rng = rng
        self.reward_model = reward_model
        self.__vectorized_rotate_obs = jax.vmap(rotate_obs, in_axes=(0, 0, 0))

    def transform(self, obs: jnp.ndarray, action: jnp.ndarray, next_obs: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray):
        rng, self.rng = jax.random.split(self.rng)
        angle_change = jax.random.uniform(rng, obs.shape[:-1], minval=-jnp.pi, maxval=jnp.pi)
        new_obs = self.__vectorized_rotate_obs(obs[:, :2], obs, angle_change)
        new_next_obs = self.__vectorized_rotate_obs(obs[:, :2], next_obs, angle_change)
        reward = self.reward_model.predict(new_next_obs, action)
        return new_obs, action, new_next_obs, reward, done

class Translation:
    """
    Translate the positions by a random delta
    Does not modify done
    """
    def __init__(self, x_range, y_range, rng, reward_model) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.rng = rng
        self.reward_model = reward_model
    
    def transform(self, obs: jnp.ndarray, action: jnp.ndarray, next_obs: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray):
        rng, self.rng = jax.random.split(self.rng)
        delta = jax.random.uniform(rng, (*obs.shape[:-1], 2),
                                   minval=jnp.array([self.x_range[0], self.y_range[0]]),
                                   maxval=jnp.array([self.x_range[1], self.y_range[1]]))
        new_obs = obs.at[..., :2].add(delta)
        new_next_obs = next_obs.at[..., :2].add(delta)
        return new_obs, action, new_next_obs, reward, done

class Compose:
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def transform(self, obs: jnp.ndarray, action: jnp.ndarray, next_obs: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray):
        data = (obs, action, next_obs, reward, done)
        for transform in self.transforms:
            data = transform.transform(*data)
        return data

class ObsJaxTransform(struct.PyTreeNode, ABC):
    @abstractmethod
    def __call__(self, obs: jnp.ndarray, rng: jax.random.KeyArray) -> jnp.ndarray:
        raise NotImplementedError
    
class JaxTranslatePos(ObsJaxTransform):
    min_x: jax.Array
    max_x: jax.Array
    min_y: jax.Array
    max_y: jax.Array

    @jax.jit
    def __call__(self, obs: jax.Array, rng: jax.random.KeyArray) -> jax.Array:
        delta = jax.random.uniform(rng, (obs.shape[0], 2,),
                                   minval=jnp.array([self.min_x, self.min_y]),
                                   maxval=jnp.array([self.max_x, self.max_y]))
        return obs.at[:,:2].add(delta)

@functools.partial(jax.vmap, in_axes=[0, 0])
def _vrotate_obs(obs, angle):
    return rotate_obs(obs[:2], obs, angle)

class JaxRotation(ObsJaxTransform):
    @jax.jit
    def __call__(self, obs: jax.Array, rng: jax.random.KeyArray) -> jax.Array:
        angle = jax.random.uniform(rng, (obs.shape[0],), minval=-jnp.pi, maxval=jnp.pi)
        return _vrotate_obs(obs, angle)

class JaxCompose(ObsJaxTransform):
    fn1: ObsJaxTransform
    fn2: ObsJaxTransform

    def __call__(self, obs: jax.Array, rng: jax.random.KeyArray) -> jax.Array:
        rng1, rng2 = jax.random.split(rng)
        return self.fn1(self.fn2(obs, rng2), rng1)

def make_compose_transform(ts: List[ObsJaxTransform]):
    if len(ts) == 1:
        return ts[0]
    composition = JaxCompose(ts[-1], ts[-2])
    for i in range(3, len(ts) + 1):
        composition = JaxCompose(ts[-i], composition)
    return composition