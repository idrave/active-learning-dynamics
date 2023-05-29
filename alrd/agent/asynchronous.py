from threading import Event, Thread
from alrd.subscriber import TopicServer
from mbse.agents.model_based.model_based_agent import ModelBasedAgent
from mbse.optimizers.sac_based_optimizer import SACOptimizer
import jax
import jax.numpy as jnp
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AsyncAgent(ModelBasedAgent):
    """
    Uses an agent to compute control sequences asynchronously.
    When act is called it returns the next action in the last computed sequence
    and supplies the observation to the agent, which uses the latest observation to
    compute the next sequence.
    """
    TIMEOUT = 1
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = None
        self.idx = None
        self.obs_srv = TopicServer('obs')
        self.act_srv = TopicServer('act')
        self.__start()

    def __start(self):
        self.step = 0
        self.idx = None
        self.stop = False
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def get_sequence(self, obs, rng, eval=False, eval_idx: int = 0):
        if isinstance(self.policy_optimizer, SACOptimizer):
            raise NotImplementedError('This is intended to work with trajectory optimizers')
        else:
            if eval:
                dim_state = obs.shape[-1]
                obs = obs.reshape(-1, dim_state)
                n_envs = obs.shape[0]
                rollout_rng, optimizer_rng = jax.random.split(rng, 2)
                rollout_rng = jax.random.split(rollout_rng, n_envs)
                optimizer_rng = jax.random.split(optimizer_rng, n_envs)
                optimize_fn = self.policy_optimizer.optimize_for_eval_fns[eval_idx]
                action_sequence, best_reward = optimize_fn(
                    dynamics_params=self.dynamics_model.model_params,
                    obs=obs,
                    key=rollout_rng,
                    optimizer_key=optimizer_rng,
                    model_props=self.dynamics_model.model_props,
                )
                actions = action_sequence
                if actions.shape[0] == 1:
                    actions = actions.squeeze(0)
            else:
                n_envs = obs.shape[0]
                rollout_rng, optimizer_rng = jax.random.split(rng, 2)
                rollout_rng = jax.random.split(rollout_rng, n_envs)
                optimizer_rng = jax.random.split(optimizer_rng, n_envs)
                action_sequence, best_reward = self.policy_optimizer.optimize_for_exploration(
                        dynamics_params=self.dynamics_model.model_params,
                        obs=obs,
                        key=rollout_rng,
                        optimizer_key=optimizer_rng,
                        model_props=self.dynamics_model.model_props,
                    )
                actions = action_sequence
            actions = actions[..., :self.action_space.shape[0]]
        return actions
    
    def _run(self):
        while not self.stop:
            step, obs, rng, eval, eval_idx = self.obs_srv.get_state(timeout=self.TIMEOUT, return_none=True)
            if obs is not None:
                actions = self.get_sequence(obs, rng, eval, eval_idx)
                self.act_srv.callback((step, actions))

    def act_in_jax(self, obs, rng, eval=False, eval_idx: int = 0):
        self.obs_srv.callback((self.step, obs, rng, eval, eval_idx))
        if self.actions is None:
            self.actions = self.act_srv.get_state(blocking=True)[1]
            self.idx = 0
        else:
            new_actions = self.act_srv.get_state(blocking=True, timeout=0, return_none=True)
            if new_actions is not None:
                logger.debug(f'Got new actions from step {new_actions[0]}. Current step is {self.step}')
                self.actions = new_actions[1]
                self.idx = self.step - new_actions[0]
        if self.idx < len(self.actions):
            action = self.actions[self.idx]
            self.idx += 1
        else:
            action = np.zeros_like(self.actions[0])
        self.step += 1
        return action
    
    def __close(self):
        self.stop = True
        self.thread.join()
    
    def reset(self):
        self.__close()
        self.obs_srv.reset()
        self.act_srv.reset()
        self.__start()
    
    def prepare_agent_for_rollout(self):
        super().prepare_agent_for_rollout()
        self.reset()

