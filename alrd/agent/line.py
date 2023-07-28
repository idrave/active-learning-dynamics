from alrd.environment.robomaster.env import VelocityControlEnv
import numpy as np

class LineAgent:
    def __init__(self, vel, steps) -> None:
        WAIT = 5
        zero = {
            VelocityControlEnv.VELOCITY: np.array([0, 0]),
            VelocityControlEnv.ANGULAR_V: np.zeros(1)
        }
        action = {
            VelocityControlEnv.VELOCITY: np.array(vel),
            VelocityControlEnv.ANGULAR_V: np.zeros(1)
        }
        rev_action = {
            VelocityControlEnv.VELOCITY: -np.array(vel),
            VelocityControlEnv.ANGULAR_V: np.zeros(1)
        }
        self.actions = steps * [action] + 2 * steps * [rev_action] + steps * [action] +  WAIT * [zero] +[None]
        self.curr = -1

    def sample_action(self, obs):
        self.curr += 1
        if self.curr == len(self.actions):
            self.curr = 0
        return self.actions[self.curr]
