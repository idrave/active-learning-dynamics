import numpy as np
from alrd.environment import RobomasterEnv
import scipy.interpolate

class RandomGPAgent:
    def __init__(self, gp, seed=0, scale=(1.,1.,1.,1.,1.), step=1):
        self.gp = gp
        self.current = None
        self.actions = None
        self.scale = np.array(scale)
        self.seed = seed
        self.step = step

    def prepare(self, num, length):
        x = np.linspace(0, length, num)[:,None]
        if self.step == 1:
            sample_x = x
        else:
            sample_x = np.linspace(0, length, (num+self.step-1)//self.step)[:,None]
        self.actions = self.gp.sample_y(sample_x, random_state=self.seed).squeeze(-1)
        if self.step != 1:
            self.actions = scipy.interpolate.interp1d(sample_x.squeeze(), self.actions, kind='cubic', axis=0)(x).squeeze(1)
        self.actions = self.actions * self.scale
        self.seed += 1
        self.current = -1
    
    def sample_action(self, obs):
        self.current = min(self.current + 1, len(self.actions) - 1)
        action = {
                RobomasterEnv.VELOCITY: self.actions[self.current,0:2].squeeze(),
                RobomasterEnv.ANGULAR_V: self.actions[self.current,2].squeeze(),
                RobomasterEnv.ARM_POSITION: np.zeros(2) # self.actions[self.current][3:5]
            }
        return action

class PiecewiseRandomGPAgent:
    def __init__(self, gp, seed=0, scale=(1.,1.,1.,1.,1.)):
        self.gp = gp
        self.current = None
        self.actions = None
        self.scale = np.array(scale)
        self.seed = seed


    def prepare(self, num, length, period):
        self.actions = np.zeros((num, 3))
        piece_duration = (length / num) * period
        for i in range(0, num, period):
            self.actions[i:i+period] = self.gp.sample_y(np.linspace(0, piece_duration, period)[:,None], n_samples=1, random_state=self.seed)[:min(period, num - i)]
            self.seed += 1
        self.actions = self.actions * self.scale
        self.current = -1
    
    def sample_action(self, obs):
        self.current = min(self.current + 1, len(self.actions) - 1)
        action = {
                RobomasterEnv.VELOCITY: self.actions[self.current,0:2],
                RobomasterEnv.ANGULAR_V: self.actions[self.current,2],
                RobomasterEnv.ARM_POSITION: self.actions[self.current,3:5]
            }
        return action