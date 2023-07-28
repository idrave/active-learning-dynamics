import jax
import jax.numpy as jnp
from alrd.utils.utils import rotate_2d_vector_jax
from mbse.utils.replay_buffer import EpisodicReplayBuffer

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
    Does not modify reward or done
    Assumes position state velocity is in global frame and actions in local frame
    """
    def __init__(self, rng) -> None:
        self.rng = rng
        self.__vectorized_rotate_obs = jax.vmap(rotate_obs, in_axes=(0, 0, 0))

    def transform(self, obs: jnp.ndarray, action: jnp.ndarray, next_obs: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray):
        rng, self.rng = jax.random.split(self.rng)
        angle_change = jax.random.uniform(rng, obs.shape[:-1], minval=-jnp.pi, maxval=jnp.pi)
        new_obs = self.__vectorized_rotate_obs(obs[:, :2], obs, angle_change)
        new_next_obs = self.__vectorized_rotate_obs(obs[:, :2], next_obs, angle_change)
        return new_obs, action, new_next_obs, reward, done