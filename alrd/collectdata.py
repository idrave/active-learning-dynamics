### Logging
import logging
logging.basicConfig(level=logging.INFO)
###
import traceback
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from gym import Env
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
import numpy as np
import time
from alrd.utils import get_timestamp_str
from alrd.environment import AbsEnv, RobomasterEnv, create_maze_env, create_robomaster_env
from alrd.agent import Agent, RandomGPAgent, KeyboardAgent, AgentType
from alrd.agent.line import LineAgent
from alrd.environment.filter import KalmanFilter
import json
import pickle
from pathlib import Path
import logging
import signal
from mbse.utils.replay_buffer import ReplayBuffer, Transition
import argparse
import yaml

logger = logging.getLogger(__file__)

def close_and_save(env: AbsEnv, buffer: ReplayBuffer, truncated_vec, info_list, output_dir):
    env.stop_robot()
    env.close()
    env.subscriber.unsubscribe()
    env.robot.close()
    print('Closed environment')
    print('Collected', buffer.current_ptr, 'transitions')
    open(output_dir/'transitions.pickle', 'wb').write(pickle.dumps(buffer)) 
    open(output_dir/'truncated.pickle', 'wb').write(pickle.dumps(truncated_vec)) 
    open(output_dir/'info.json', 'w').write(json.dumps(info_list))
    print(f'Output {output_dir}')

class HandleSignal:
    def __init__(self, handler):
        self.handler = handler
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, signum, frame):
        logging.info('Closed with sigint')
        self.handler()

def init_filter(freq, use_acc):
    # TODO: deprecated, must be updated
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

def collect_data_buffer(agent: Agent, env: AbsEnv, buffer, truncated_vec, info_list, max_steps, buffer_size):
    num_points = int(max_steps)
    obs_shape = (num_points,) + env.observation_space.shape
    action_space = (num_points,) + env.action_space.shape
    started = False
    for step in range(buffer_size):
        if not started:
            count = 0
            obs, info = env.reset()
            start = time.time()
            info_list.append(info)
            obs_vec = np.zeros(obs_shape)
            action_vec = np.zeros(action_space)
            reward_vec = np.zeros((num_points,))
            next_obs_vec = np.zeros(obs_shape)
            done_vec = np.zeros((num_points,))
            started = True
        action = agent.act(obs)
        obs_vec[count] = obs
        action_vec[count] = action
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward_vec[count] = reward
        next_obs_vec[count] = next_obs
        done_vec[count] = terminated
        truncated_vec[step] = truncated
        info_list.append(info)
        count += 1
        obs = next_obs
        if terminated or truncated:
            transitions = Transition(
                obs=obs_vec,
                action=action_vec,
                reward=reward_vec,
                next_obs=next_obs_vec,
                done=done_vec,
            )
            buffer.add(transitions)
            env.stop_robot()
            started = False
            print('Terminated %s. Truncated %s' % (terminated, truncated))
            print('Episode length %d. Elapsed time %f. Average step time %f' % (count, time.time() - start, (time.time() - start)/count))


if __name__ == '__main__':
    logger.info('Collecting data')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_steps', default=100000, type=int, help='Number of steps to record')
    parser.add_argument('-e', '--episode_len', default=None, type=int, help='Maximum episode length')
    parser.add_argument('-f', '--freq', default=50, type=int, help='Frequency of the environment')
    parser.add_argument('-a', '--agent', default=AgentType.KEYBOARD, type=AgentType, help='Agent to use', choices=[a for a in AgentType])
    # Agent arguments
    agent_parser = parser.add_argument_group('Agent arguments')
    agent_parser.add_argument('--seed', default=0, type=int, help='Seed for random action sampling')
    agent_parser.add_argument('--xy_speed', default=0.5, type=float, help='Speed of the agent in the xy plane')
    agent_parser.add_argument('--a_speed', default=120, type=float, help='Angular speed of the agent')
    agent_parser.add_argument('--length_scale', default=1, type=float, help='Length scale of the GP agent')
    agent_parser.add_argument('--noise', default=1e-3, type=float, help='Noise of the GP agent')
    agent_parser.add_argument('--gp_undersample', default=4, type=int, help='Undersampling factor of the GP agent')
    args = parser.parse_args()
    output_dir = Path('output')/'data'/('%s'%get_timestamp_str())
    output_dir.mkdir()
    yaml.dump(vars(args), open(output_dir/'args.yaml', 'w'))
    # Maze coordinates
    p0 = np.array([-0.65, -0.77])
    diffs = np.array([
        [0,1.52],
        [1.6,0],
        [0,1.6],
        [2.1,0],
        [0,-1.6],
        [-1.5,0],
        [0,-1.6],
        [1.5,0],
        [0,-1.6],
        [-2.1,0],
        [0,1.6],
    ])
    points = [p0]
    for d in diffs:
        points.append(points[-1]+d)
    coords = np.stack(points)
    use_filter = False
    transforms = []
    if use_filter:
        transforms.append(init_filter(50, False))
    env = create_maze_env(coordinates=coords, margin=0.30, freq=args.freq, transforms=transforms)
    env = RescaleAction(env, min_action=-1, max_action=1)
    max_episode_steps = args.episode_len
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    agent = args.agent(args)
    sequences = []
    sub_logs = []
    buffer = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        normalize=True,
        action_normalize=False,
        learn_deltas=True
    )
    truncated_vec = np.zeros((args.n_steps,))
    info_list = []
    def finish():
        close_and_save(env, buffer, truncated_vec, info_list, output_dir)        
    handler = HandleSignal(finish)
    try:
        collect_data_buffer(
            agent, env, buffer, truncated_vec, info_list, max_episode_steps, args.n_steps)
    except Exception as e:
        traceback.print_exc()
    finally:
        finish()
