import numpy as np
from alrd.agent.absagent import Agent
import scipy.interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import threading
from queue import Queue, Empty

def create_rbf_gp_agent(length_scale, noise, scale, max_steps, freq=50, seed=0, sample=1):
    kernel = RBF(length_scale=length_scale, length_scale_bounds='fixed') + WhiteKernel(noise_level=noise, noise_level_bounds='fixed')
    gp = GaussianProcessRegressor(kernel, random_state=seed)
    gp.fit(np.zeros((1,1)), np.zeros_like(np.array(scale))[None,:])
    return RandomGPAgent(gp, scale, max_steps, freq=freq, seed=seed, sample=sample)

def create_async_rbf_gp_agent(length_scale, noise, scale, max_steps, freq=50, seed=0, sample=1, queue_sz=1):
    kernel = RBF(length_scale=length_scale, length_scale_bounds='fixed') + WhiteKernel(noise_level=noise, noise_level_bounds='fixed')
    gp = GaussianProcessRegressor(kernel, random_state=seed)
    return AsyncGPAgent(gp, scale, max_steps, freq=freq, seed=seed, sample=sample, queue_sz=queue_sz)

class RandomGPAgent(Agent):
    def __init__(self, gp: GaussianProcessRegressor, scale, max_steps, freq=50, seed=0, sample=1):
        self.gp = gp
        gp.fit(np.zeros((1,1)), np.zeros_like(scale)[None])
        self.current = None
        self.actions = None
        self.scale = np.array(scale)
        self.seed = seed
        self.max_steps = max_steps
        self.sample = sample
        self.period = 1/freq

    def prepare(self):
        x = np.linspace(0, self.max_steps*self.period, self.max_steps)[:,None]
        if self.sample == 1:
            sample_x = x
        else:
            sample_x = np.linspace(0, self.max_steps*self.period, (self.max_steps+self.sample-1)//self.sample)[:,None]
        actions = self.gp.sample_y(sample_x, random_state=self.seed).squeeze(-1)
        if self.sample != 1:
            actions = scipy.interpolate.interp1d(sample_x.squeeze(), actions, kind='cubic', axis=0)(x).squeeze(1)
        actions = actions * self.scale
        self.seed += 1
        return actions
    
    def act(self, obs):
        if self.actions is None:
            self.actions = self.prepare()
            self.current = -1
        self.current = min(self.current + 1, len(self.actions) - 1)
        return np.tanh(self.actions[self.current])
    
    def reset(self):
        self.actions = None

class AsyncGPAgent(RandomGPAgent):
    def __init__(self, gp, scale, max_steps, freq=50, seed=0, sample=1, queue_sz=1):
        super().__init__(gp, scale, max_steps, freq, seed, sample)
        self.q = Queue(maxsize=queue_sz)
        self.closed = False
        self.thread = threading.Thread(target=self.__start, daemon=True)
        self.thread.start()
    
    def __start(self):
        while not self.closed:
            self.q.put(self.prepare())

    def act(self, obs):
        if self.actions is None or self.current == len(self.actions) - 1:
            try:
                self.actions = self.q.get(timeout=self.period)
                self.current = -1
            except Empty:
                pass
        self.current = min(self.current + 1, len(self.actions) - 1)
        return np.tanh(self.actions[self.current])

    def join(self):
        self.closed = True
        self.q.get()
        self.join()

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