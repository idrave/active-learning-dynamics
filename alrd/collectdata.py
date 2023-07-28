### Logging
import logging

from alrd.environment.spot.spotgym import SpotGym

logging.basicConfig(level=logging.INFO)
import argparse
import json
import logging
import pickle
import signal
import time
###
import traceback
from pathlib import Path
from typing import Union

import jax
import numpy as np
import tqdm
import yaml
from alrd.agent import (Agent, AgentType, 
                        SpotAgentEnum, create_spot_agent)
from alrd.agent.repeat import RepeatAgent
from alrd.environment import (BaseRobomasterEnv, create_robomaster_env,
                              create_spot_env)
from alrd.environment.filter import KalmanFilter
from alrd.environment.spot.spotgym import ResetEnum
from alrd.utils import get_timestamp_str
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.time_limit import TimeLimit

from mbse.utils.replay_buffer import (EpisodicReplayBuffer, ReplayBuffer,
                                      Transition)

logger = logging.getLogger(__file__)

def close_and_save(env: BaseRobomasterEnv, buffer: ReplayBuffer, truncated_vec, info_list, output_dir):
    env.close()
    print('Closed environment')
    print('Collected', buffer.current_ptr, 'transitions')
    open(output_dir/'transitions.pickle', 'wb').write(pickle.dumps(buffer)) 
    open(output_dir/'truncated.pickle', 'wb').write(pickle.dumps(truncated_vec)) 
    open(output_dir/'info.json', 'w').write(json.dumps(info_list))
    #open(output_dir/'state_log.json', 'w').write(json.dumps(env.get_state_log()))
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

def collect_data_buffer(agent: Agent, env: Union[BaseRobomasterEnv, SpotGym], buffer, truncated_vec, info_list, max_steps,
                        num_steps, num_episodes, avoid_pose_reset, use_tqdm):
    num_points = max_steps
    obs_space =  env.observation_space.shape
    action_space =  env.action_space.shape
    started = False
    episode_count = 0
    step = 0
    if use_tqdm:
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps)
        else:
            pbar = tqdm.tqdm(total=num_episodes)
    while (num_episodes is None or episode_count < num_episodes) and (num_steps is None or step < num_steps):
        if not started:
            count = 0
            options = {}
            if avoid_pose_reset:
                options['action'] = ResetEnum.STAND
            obs, info = env.reset(options=options)
            if obs is None:
                logger.info("Reset returned None state. Stopping collection.")
                return
            agent.reset()
            start = time.time()
            info_list.append(info)
            obs_vec = np.zeros((num_points,) + obs_space)
            action_vec = np.zeros((num_points,) + action_space)
            reward_vec = np.zeros((num_points,))
            next_obs_vec = np.zeros((num_points,) + obs_space)
            done_vec = np.zeros((num_points,))
            started = True
        agent_start = time.time()
        action = agent.act(obs)
        agent_time = time.time() - agent_start
        if action is not None:
            next_obs, reward, terminated, truncated, info = env.step(action)
            if next_obs is not None:
                obs_vec[count] = obs
                action_vec[count] = action
                reward_vec[count] = reward
                next_obs_vec[count] = next_obs
                done_vec[count] = terminated
                truncated_vec[step] = truncated
                info['agent_time'] = agent_time
                info_list.append(info)
                count += 1
                step += 1
                if use_tqdm and num_steps is not None:
                    pbar.update(1)
        if action is None or terminated or truncated or step == num_steps:
            #env.stop_robot()
            started = False
            if count > 0:
                transitions = Transition(
                    obs=obs_vec[:count],
                    action=action_vec[:count],
                    reward=reward_vec[:count],
                    next_obs=next_obs_vec[:count],
                    done=done_vec[:count],
                )
                buffer.add(transitions)
                print('Terminated %s. Truncated %s' % (terminated, truncated))
                print('Episode length %d. Elapsed time %f. Average step time %f' % (count, time.time() - start, (time.time() - start)/count))
                episode_count += 1
                if use_tqdm and num_episodes is not None:
                    pbar.update(1)
        else:
            obs = next_obs
    env.reset()

def add_common_args(main_parser: argparse.ArgumentParser):
    main_parser.add_argument('--tag', type=str, default='data')
    main_parser.add_argument('--tqdm', action='store_true')
    main_parser.add_argument('-n', '--n_steps', default=None, type=int, help='Number of steps to record')
    main_parser.add_argument('-ne', '--num_episodes', default=None, type=int, help='Number of episodes to record')
    main_parser.add_argument('-e', '--episode_len', default=None, type=int, help='Maximum episode length')
    main_parser.add_argument('-f', '--freq', default=10, type=int, help='Frequency at which commands are supplied to the environment')
    main_parser.add_argument('-o', '--output', default='output', type=str, help='Output directory')
    main_parser.add_argument('--noactnorm', action='store_true', help='Set action normalization to False in replay buffer')
    main_parser.add_argument('--seed', default=0, type=int, help='Randomization seed')

def add_robomaster_parser(parser: argparse.ArgumentParser):
    # Environment arguments
    add_common_args(parser)
    env_parser = parser.add_argument_group('Robomaster environment arguments')
    env_parser.add_argument('--poscontrol', action='store_true')
    env_parser.add_argument('--raw_pos', action='store_true', help='Whether to use raw positon data instead of estimating from accelerometer')
    env_parser.add_argument('--square', action='store_true', help='Square environment')
    env_parser.add_argument('--cossin', action='store_true')
    env_parser.add_argument('--noangle', action='store_true')
    env_parser.add_argument('--novelocity', action='store_true')
    env_parser.add_argument('--repeat_action', default=None, type=int, help='Number of times to repeat the action. Reduces the required control frequency')
    env_parser.add_argument('--global_frame', action='store_true', help='Use global frame for environment and agent')
    env_parser.add_argument('--margin', type=float, default=0.3)
    env_parser.add_argument('--slide', action='store_true')
    # Agent arguments
    agent_parser = parser.add_argument_group('Agent arguments')
    agent_parser.add_argument('-a', '--agent', default=AgentType.KEYBOARD, type=AgentType, help='Agent to use', choices=[a for a in AgentType])
    agent_parser.add_argument('--xy_speed', default=0.5, type=float, help='Speed of the agent in the xy plane')
    agent_parser.add_argument('--a_speed', default=120, type=float, help='Angular speed of the agent')
    agent_parser.add_argument('--length_scale', default=1, type=float, help='Length scale of the GP agent')
    agent_parser.add_argument('--noise', default=1e-3, type=float, help='Noise of the GP agent')
    agent_parser.add_argument('--gp_undersample', default=4, type=int, help='Undersampling factor of the GP agent')
    agent_parser.add_argument('--agent_checkpoint', default=None, type=str, help='Path to SAC agent checkpoint')
    agent_parser.add_argument('--model_checkpoint', default=None, type=str, help='Path to model checkpoint')
    agent_parser.add_argument('--horizon', default=60, type=int, help='Horizon of the MPC agent')

def add_spot_parser(parser: argparse.ArgumentParser):
    add_common_args(parser)
    parser.add_argument('config', type=str, help='Environment configuration file')
    parser.add_argument('--monitor', type=int, default=30, help='Frequency at which the environment checks if the robot is within boundaries (default: 30))')
    parser.add_argument('-a', '--agent', default=SpotAgentEnum.KEYBOARD.value, type=str, help='Agent to use', choices=[a.value for a in SpotAgentEnum])
    parser.add_argument('--smoothing', default=None, type=float, help='Smoothing factor for the actions')
    parser.add_argument('--optimizer_checkpoint', default=None, type=str, help='Path to optimizer checkpoint')
    parser.add_argument('--query_goal', action='store_true', help='Whether to query the goal from the user at every reset')
    parser.add_argument('--avoid_pose_reset', action='store_true', help="Whether to reset the robot's pose only when necessary and not at every episode")

def collect_data(args):
    output_dir = Path(args.output)/('%s-%s'%(args.tag,get_timestamp_str()))
    output_dir.mkdir(parents=True)
    yaml.dump(vars(args), open(output_dir/'args.yaml', 'w'))
    rng = jax.random.PRNGKey(args.seed)
    if args.robot == 'robomaster':
        env = create_robomaster_env(
            poscontrol=args.poscontrol,
            estimate_from_acc=not args.raw_pos,
            margin=args.margin,
            freq=args.freq,
            slide_wall=args.slide,
            global_frame=args.global_frame,
            cossin=args.cossin,
            noangle=args.noangle,
            novelocity=args.novelocity,
            repeat_action=None, # args.repeat_action,
            square=args.square,
            xy_speed=args.xy_speed,
            a_speed=args.a_speed
        )
        rng, agent_rng = jax.random.split(rng)
        args.agent_rng = agent_rng
        args.reward_model = env.unwrapped.reward
        agent = args.agent(args)
        if args.repeat_action is not None:
            agent = RepeatAgent(agent, args.repeat_action)
        buffer = ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            normalize=True,
            action_normalize=not args.noactnorm,
            learn_deltas=True
        )
    elif args.robot == 'spot':
        agent_type=SpotAgentEnum(args.agent)
        config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
        env = create_spot_env(
            config=config,
            cmd_freq=args.freq,
            monitor_freq=args.monitor,
            log_dir=output_dir,
            query_goal=args.query_goal
        )
        rng, agent_rng = jax.random.split(rng)
        agent = create_spot_agent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            agent_type=agent_type,
            optimizer_path=args.optimizer_checkpoint,
            smoothing_coeff=args.smoothing,
            rng=agent_rng
        )
        buffer = EpisodicReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            normalize=True,
            action_normalize=not args.noactnorm,
            learn_deltas=True
        )
        env.start()
    else:
        raise NotImplementedError(f'Robot {args.robot} not implemented')
    print('env', type(env), 'obs', env.observation_space.shape)
    env = RescaleAction(env, min_action=-1, max_action=1)
    max_episode_steps = args.episode_len
    if max_episode_steps is not None:
        print(max_episode_steps)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    truncated_vec = np.zeros((args.n_steps,)) # TODO need to fix this for when number of episodes is specified
    info_list = []
    try:
        collect_data_buffer(
            agent, env, buffer, truncated_vec, info_list, max_episode_steps, args.n_steps, args.num_episodes, args.avoid_pose_reset, use_tqdm=args.tqdm)
    except KeyboardInterrupt:
        print('Collection interrupted!')
    except Exception as e:
        traceback.print_exc()
    finally:
        close_and_save(env, buffer, truncated_vec, info_list, output_dir)        

if __name__ == '__main__':
    logger.info('Collecting data')
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Robot', dest='robot', help='Robot on which to collect data', required=True)
    add_robomaster_parser(subparsers.add_parser('robomaster'))
    add_spot_parser(subparsers.add_parser('spot'))
    args = parser.parse_args()
    assert args.n_steps is not None or args.num_episodes is not None, 'Either number of steps or number of episodes must be specified'
    assert args.n_steps is None or args.num_episodes is None, 'Either number of steps or number of episodes must be specified, not both'
    collect_data(args)