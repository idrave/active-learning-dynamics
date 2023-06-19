### Logging
import logging
logging.basicConfig(level=logging.INFO)
###
import traceback
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.rescale_action import RescaleAction
import numpy as np
import time
from alrd.utils import get_timestamp_str, convert_to_cos_sin
from alrd.environment import BaseRobomasterEnv, create_robomaster_env
from alrd.environment.maze import MazeGoalVelocityEnv, MazeGoalPositionEnv
from alrd.agent import Agent, RandomGPAgent, KeyboardAgent, AgentType
from alrd.agent.repeat import RepeatAgent
from alrd.environment.filter import KalmanFilter
import json
import pickle
from pathlib import Path
import logging
import signal
from mbse.utils.replay_buffer import ReplayBuffer, Transition
import argparse
import yaml
import jax
from alrd.environment.wrappers import CosSinObsWrapper, RemoveAngleActionWrapper, KeepObsWrapper, RepeatActionWrapper, GlobalFrameActionWrapper
import tqdm

logger = logging.getLogger(__file__)

def close_and_save(env: BaseRobomasterEnv, buffer: ReplayBuffer, truncated_vec, info_list, output_dir):
    env.stop_robot()
    env.close()
    env.subscriber.unsubscribe()
    env.robot.close()
    print('Closed environment')
    print('Collected', buffer.current_ptr, 'transitions')
    open(output_dir/'transitions.pickle', 'wb').write(pickle.dumps(buffer)) 
    open(output_dir/'truncated.pickle', 'wb').write(pickle.dumps(truncated_vec)) 
    open(output_dir/'info.json', 'w').write(json.dumps(info_list))
    open(output_dir/'state_log.json', 'w').write(json.dumps(env.get_state_log()))
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

def collect_data_buffer(agent: Agent, env: BaseRobomasterEnv, buffer, truncated_vec, info_list, max_steps, buffer_size, use_tqdm):
    num_points = max_steps
    obs_space =  env.observation_space.shape
    action_space =  env.action_space.shape
    started = False
    it = range(buffer_size)
    if use_tqdm:
        it = tqdm.tqdm(it)
    for step in it:
        if not started:
            count = 0
            obs, info = env.reset()
            agent.reset()
            start = time.time()
            info_list.append(info)
            obs_vec = np.zeros((num_points,) + obs_space)
            action_vec = np.zeros((num_points,) + action_space)
            reward_vec = np.zeros((num_points,))
            next_obs_vec = np.zeros((num_points,) + obs_space)
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
                obs=obs_vec[:count],
                action=action_vec[:count],
                reward=reward_vec[:count],
                next_obs=next_obs_vec[:count],
                done=done_vec[:count],
            )
            buffer.add(transitions)
            env.stop_robot()
            started = False
            print('Terminated %s. Truncated %s' % (terminated, truncated))
            print('Episode length %d. Elapsed time %f. Average step time %f' % (count, time.time() - start, (time.time() - start)/count))


if __name__ == '__main__':
    logger.info('Collecting data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument('-n', '--n_steps', default=100000, type=int, help='Number of steps to record')
    parser.add_argument('-e', '--episode_len', default=None, type=int, help='Maximum episode length')
    parser.add_argument('-f', '--freq', default=10, type=int, help='Frequency of the environment')
    parser.add_argument('-a', '--agent', default=AgentType.KEYBOARD, type=AgentType, help='Agent to use', choices=[a for a in AgentType])
    # Environment arguments
    parser.add_argument('--poscontrol', action='store_true')
    parser.add_argument('--raw_pos', action='store_true', help='Whether to use raw positon data instead of estimating from accelerometer')
    parser.add_argument('--square', action='store_true', help='Square environment')
    parser.add_argument('--cossin', action='store_true')
    parser.add_argument('--noangle', action='store_true')
    parser.add_argument('--novelocity', action='store_true')
    parser.add_argument('--repeat_action', default=None, type=int, help='Number of times to repeat the action. Reduces the required control frequency')
    parser.add_argument('--global_frame', action='store_true', help='Use global frame for environment and agent')
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--slide', action='store_true')
    # Agent arguments
    agent_parser = parser.add_argument_group('Agent arguments')
    agent_parser.add_argument('--seed', default=0, type=int, help='Seed for random action sampling')
    agent_parser.add_argument('--xy_speed', default=0.5, type=float, help='Speed of the agent in the xy plane')
    agent_parser.add_argument('--a_speed', default=120, type=float, help='Angular speed of the agent')
    agent_parser.add_argument('--length_scale', default=1, type=float, help='Length scale of the GP agent')
    agent_parser.add_argument('--noise', default=1e-3, type=float, help='Noise of the GP agent')
    agent_parser.add_argument('--gp_undersample', default=4, type=int, help='Undersampling factor of the GP agent')
    agent_parser.add_argument('--agent_checkpoint', default=None, type=str, help='Path to SAC agent checkpoint')
    agent_parser.add_argument('--model_checkpoint', default=None, type=str, help='Path to model checkpoint')
    agent_parser.add_argument('--horizon', default=60, type=int, help='Horizon of the MPC agent')
    args = parser.parse_args()
    output_dir = Path('output')/'data'/('%s-%s'%(args.tag,get_timestamp_str()))
    output_dir.mkdir()
    yaml.dump(vars(args), open(output_dir/'args.yaml', 'w'))
    env = create_robomaster_env(
        args.poscontrol,
        not args.raw_pos,
        args.margin,
        args.freq,
        args.slide,
        args.global_frame,
        args.cossin,
        args.noangle,
        args.novelocity,
        None, # args.repeat_action,
        args.square,
        args.xy_speed,
        args.a_speed
    )
    print('env', type(env), 'obs', env.observation_space.shape)
    env = RescaleAction(env, min_action=-1, max_action=1)
    max_episode_steps = args.episode_len
    if max_episode_steps is not None:
        print(max_episode_steps)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    args.agent_rng = jax.random.PRNGKey(args.seed)
    args.reward_model = env.unwrapped.reward
    agent = args.agent(args)
    if args.repeat_action is not None:
        agent = RepeatAgent(agent, args.repeat_action)
    sequences = []
    sub_logs = []
    buffer = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        normalize=True,
        action_normalize=True,
        learn_deltas=True
    )
    truncated_vec = np.zeros((args.n_steps,))
    info_list = []
    def finish():
        close_and_save(env, buffer, truncated_vec, info_list, output_dir)        
    handler = HandleSignal(finish)
    try:
        collect_data_buffer(
            agent, env, buffer, truncated_vec, info_list, max_episode_steps,  args.n_steps, use_tqdm=args.tqdm)
    except Exception as e:
        traceback.print_exc()
    finally:
        finish()
