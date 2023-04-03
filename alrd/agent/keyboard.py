from alrd.environment.env import VelocityControlEnv
from alrd.ui import KeyboardListener
import numpy as np

class KeyboardAgent:
    def __init__(self, xy_speed, a_speed) -> None:
        self.listener = KeyboardListener()
        self.xy_speed = xy_speed
        self.a_speed = a_speed
        self.cmds = {
            'w': (1, 0, 0),
            'a': (0, 1, 0),
            's': (-1, 0, 0),
            'd': (0, -1, 0),
            'q': (0, 0, 1),
            'e': (0, 0, -1)
        }

    def sample_action(self, obs):
        pressed = self.listener.which_pressed(self.cmds.keys())
        action = {
            VelocityControlEnv.VELOCITY: np.array([0, 0]),
            VelocityControlEnv.ANGULAR_V: 0
        }
        for key in pressed:
            action[VelocityControlEnv.VELOCITY] += self.cmds[key][0:2]
            action[VelocityControlEnv.ANGULAR_V] += self.cmds[key][2]
        action[VelocityControlEnv.VELOCITY] = self.xy_speed * action[VelocityControlEnv.VELOCITY] / np.linalg.norm(action[VelocityControlEnv.VELOCITY])
        action[VelocityControlEnv.ANGULAR_V] = self.a_speed * action[VelocityControlEnv.ANGULAR_V]
        return action