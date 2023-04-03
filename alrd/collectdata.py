import traceback
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
import numpy as np
import time
from alrd.utils import get_timestamp_str
from alrd.environment import RobomasterEnv, create_maze_env
from alrd.agent import RandomGPAgent, KeyboardAgent
import json
import pickle
from pathlib import Path
import logging
import signal

class HandleSignal:
    def __init__(self, env):
        self.env = env
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logging.info('Closed with sigint')
        self.env.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Collecting data')
    datatime = 60.
    collected = 0.
    x_min, x_max, y_min, y_max = -1.5, 1.5, -1.5, 1.5
    coords = np.array([
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min)
    ])
    robomaster_env = create_maze_env(coordinates=coords, margin=0.20, freq=50)
    env = RecordEpisodeStatistics(robomaster_env)
    handler = HandleSignal(env)
    sample_freq = 50
    episode_seconds = 60
    total = sample_freq * episode_seconds
    max_episode_steps = total + 1
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    kernel =  RBF(length_scale=1., length_scale_bounds='fixed')
    kernel += WhiteKernel(noise_level=1e-2, noise_level_bounds='fixed')
    # gpr = GaussianProcessRegressor(kernel=kernel).fit(X=np.array([[0]]), y=np.zeros((1,5)))
    # var_scale = (1,1,100.,20.,20.)
    # agent = RandomGPAgent(gpr, scale=var_scale, step=4)
    agent = KeyboardAgent(1., 120)
    sequences = []
    sub_logs = []
    try:
        while collected < datatime:
            episode = []
            testaction = {
                RobomasterEnv.VELOCITY: np.array([1.0, -0.5]),
                RobomasterEnv.ANGULAR_V: 15.,
                # RobomasterEnv.ARM_POSITION: np.zeros(2)
            }
            obs, info = env.reset()
            #print('preparing agent')
            #agent.prepare(total, episode_seconds)
            start = time.time()
            done = False
            count = 0
            while not done:
                action = agent.sample_action(obs)
                # action = testaction
                new_obs, reward, terminated, truncated, new_info = env.step(action)
                episode.append((obs, info, action, new_obs, new_info, terminated, truncated))
                obs = new_obs
                info = new_info
                count += 1
                done = terminated or truncated
            print('Terminated %s. Truncated %s' % (terminated, truncated))
            #print('Episode length %d. Elapsed time %f. Average step time %f' % (info['episode']['l'], info['episode']['t'], info['episode']['t']/info['episode']['l']))
            print('Episode length %d. Elapsed time %f. Average step time %f' % (count, time.time() - start, (time.time() - start)/count))
            collected += time.time() - start
            sequences.append(episode)
            sub_logs.append(robomaster_env.get_subscriber_log())
        env.reset()
    except Exception as e:
        traceback.print_exc()
        env.robot.chassis.drive_speed(0, 0, 0)

    env.close()
    output_dir = Path('output')/'data'/('%s'%get_timestamp_str())
    output_dir.mkdir()
    open(output_dir/'transitions.pickle', 'wb').write(pickle.dumps(sequences)) 
    open(output_dir/'subscriber.json', 'w').write(json.dumps(sub_logs))
