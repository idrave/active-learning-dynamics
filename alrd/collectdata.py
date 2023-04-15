import traceback
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
import numpy as np
import time
from alrd.utils import get_timestamp_str
from alrd.environment import RobomasterEnv, create_maze_env, create_robomaster_env
from alrd.agent import RandomGPAgent, KeyboardAgent
from alrd.agent.line import LineAgent
from alrd.environment.filter import KalmanFilter
import json
import pickle
from pathlib import Path
import logging
import signal

logger = logging.getLogger(__file__)

class HandleSignal:
    def __init__(self, env, sequences, sub_logs):
        self.env = env
        self.sequences = sequences
        self.sub_logs = sub_logs
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logging.info('Closed with sigint')
        self.sub_logs.append(self.env.get_subscriber_log())
        self.env.close()
        print('Closed environment')
        output_dir = Path('output')/'data'/('%s'%get_timestamp_str())
        output_dir.mkdir()
        open(output_dir/'transitions.pickle', 'wb').write(pickle.dumps(self.sequences)) 
        open(output_dir/'subscriber.json', 'w').write(json.dumps(self.sub_logs))
        print(f'Output {output_dir}')

def init_filter(freq, use_acc):
    if use_acc:
        return KalmanFilter(1/freq,
                            pred_cov = np.diag([1, 1, 4]),
                            obs_cov = np.diag([8, 1, 1]),
                            init_cov = 0.01 * np.diag([1, 1, 1]),
                            use_acc=True)
    else:
        return KalmanFilter(1/freq,
                            pred_cov = np.diag([1, 4]),
                            obs_cov = np.diag([4, 1]),
                            init_cov = 0.01 * np.diag([1, 1]),
                            use_acc=False, pred_displ=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Collecting data')
    datatime = 4
    collected = 0.
    x_min, x_max, y_min, y_max = -3.+0.19, 3.-0.19, -3.+0.12, 3.-0.12
    coords = np.array([
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min)
    ])
    # robomaster_env = create_robomaster_env(x_min, x_max, y_min, y_max, freq=50)
    use_filter = False
    transforms = []
    if use_filter:
        transforms.append(init_filter(50, False))
    robomaster_env = create_maze_env(coordinates=coords, margin=0.10, freq=50, transforms=transforms)
    env = robomaster_env
    sample_freq = 50
    episode_seconds = 60
    total = sample_freq * episode_seconds
    max_episode_steps = None
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    kernel =  RBF(length_scale=1., length_scale_bounds='fixed')
    kernel += WhiteKernel(noise_level=1e-2, noise_level_bounds='fixed')
    # gpr = GaussianProcessRegressor(kernel=kernel).fit(X=np.array([[0]]), y=np.zeros((1,5)))
    # var_scale = (1,1,100.,20.,20.)
    # agent = RandomGPAgent(gpr, scale=var_scale, step=4)
    # agent = KeyboardAgent(1., 120)
    agent = LineAgent((0.7, 0.7), 50)
    sequences = []
    sub_logs = []
    handler = HandleSignal(env, sequences, sub_logs)
    try:
        while collected < datatime:
            episode = []
            testaction = {
                RobomasterEnv.VELOCITY: np.array([1.0, -0.5]),
                RobomasterEnv.ANGULAR_V: 15.,
                # RobomasterEnv.ARM_POSITION: np.zeros(2)
            }
            print('calling reset')
            obs, info = env.reset(seed=None)
            #print('preparing agent')
            #agent.prepare(total, episode_seconds)
            start = time.time()
            done = False
            count = 0
            while not done:
                action = agent.sample_action(obs)
                #action = None
                new_obs, reward, terminated, truncated, new_info = env.step(action)
                episode.append((obs, info, action, new_obs, new_info, terminated, truncated))
                obs = new_obs
                info = new_info
                count += 1
                done = terminated or truncated
            print('Terminated %s. Truncated %s' % (terminated, truncated))
            #print('Episode length %d. Elapsed time %f. Average step time %f' % (info['episode']['l'], info['episode']['t'], info['episode']['t']/info['episode']['l']))
            print('Episode length %d. Elapsed time %f. Average step time %f' % (count, time.time() - start, (time.time() - start)/count))
            print('Last obs position: %s' % obs['position'])
            collected += time.time() - start
            sequences.append(episode)
            sub_logs.append(robomaster_env.get_subscriber_log())
        env.reset()
    except Exception as e:
        traceback.print_exc()
        env.robot.chassis.drive_speed(0, 0, 0)

    env.close()
    print('Closed environment')
    output_dir = Path('output')/'data'/('%s'%get_timestamp_str())
    output_dir.mkdir()
    open(output_dir/'transitions.pickle', 'wb').write(pickle.dumps(sequences)) 
    open(output_dir/'subscriber.json', 'w').write(json.dumps(sub_logs))
    print(f'Output {output_dir}')
