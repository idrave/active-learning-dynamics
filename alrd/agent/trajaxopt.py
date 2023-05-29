from trajax.optimizers import ILQRHyperparams
from mbse.optimizers.trajax_trajectory_optimizer import TraJaxTO
from mbse.models.bayesian_dynamics_model import BayesianDynamicsModel
from alrd.agent.absagent import Agent
import jax

class TraJaxOptAgent(Agent):
    MODEL_INDEX = 0
    def __init__(self, model: BayesianDynamicsModel, optimizer: TraJaxTO, rng=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.actions = None
        self.rng = rng
        self.idx = None
    
    @staticmethod
    def create(model: BayesianDynamicsModel, action_dim, horizon, rng=None):
        optimizer = TraJaxTO(
            horizon=horizon,
            action_dim=action_dim,
            dynamics_model_list=[model]
        )
        return TraJaxOptAgent(model, optimizer, rng=rng)

    def act(self, obs):
        if self.actions is None:
            if self.rng is None:
                key = None
                opt_key = None
            else:
                self.rng, key, opt_key = jax.random.split(self.rng, 3)
                key = key[None,:]
                opt_key = opt_key[None,:]
            sequence, _ = self.optimizer.optimize_action_sequence_for_evaluation(
                self.MODEL_INDEX, self.model.model_params, obs[None,:], key, opt_key, self.model.model_props
            )
            self.actions = sequence[0]
            self.idx = 0
        action = self.actions[self.idx]
        self.idx = max(self.idx + 1, len(self.actions) - 1)
        return action
    
    def reset(self):
        self.actions = None
        self.idx = None