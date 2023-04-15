import pykalman
import numpy as np
from alrd.environment.env import AbsEnv
import copy
import logging
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

def assert_param_shape(name, param, shape):
    assert param.shape == shape, f"Expected shape {shape} for {name}, got {param.shape}"

class KalmanFilter:
    def __init__(self, pred_cov, obs_cov, init_cov, use_acc=False, pred_displ=False, drift=False) -> None:
        self.use_acc = use_acc
        self.n_dim_state = 3 if use_acc else 2
        self.n_dim_obs = 3 if use_acc else 2
        if drift:
            self.n_dim_state += 1
        assert_param_shape("pred_cov", pred_cov, (self.n_dim_state, self.n_dim_state))
        assert_param_shape("obs_cov", obs_cov, (self.n_dim_obs, self.n_dim_obs))
        assert_param_shape("init_cov", init_cov, (self.n_dim_state, self.n_dim_state))
        assert not (pred_displ and drift), "Cannot predict displacement and drift at the same time"
        self.pred_displ = pred_displ
        self.drift = drift
        self.filter = pykalman.KalmanFilter(
            transition_matrices=None,
            observation_matrices=self.get_observation_matrix(),
            transition_covariance=pred_cov,
            observation_covariance=obs_cov,
            initial_state_mean=np.zeros(self.n_dim_state),
            initial_state_covariance=init_cov)
        self.curr_mean_x = self.filter.initial_state_mean
        self.curr_mean_y = self.filter.initial_state_mean
        self.curr_cov_x = init_cov
        self.curr_cov_y = init_cov
        # Only used for displacement
        self.last_pos_obs = np.array([self.filter.initial_state_mean[0], self.filter.initial_state_mean[0]])
        self.last_pos_pred = np.array([self.filter.initial_state_mean[0], self.filter.initial_state_mean[0]])

    def get_transition_matrix(self, T):
        if self.drift:
            return np.array(
                [[1, T, 0.5 * T ** 2, 0], [0, 1, T, 0], [0, 0, 1, 0], [0, 0, 0, 1]] if self.use_acc else
                [[1, T, 0], [0, 1, 0], [0, 0, 1]]
            )
        elif self.pred_displ:
            return np.array(
                [[0, T, 0.5 * T ** 2], [0, 1, T], [0, 0, 1]] if self.use_acc else
                [[0, T], [0, 1]]
            )
        else:
            return np.array(
                [[1, T, 0.5 * T ** 2], [0, 1, T], [0, 0, 1]] if self.use_acc else
                [[1, T], [0, 1]]
            )

    def get_observation_matrix(self):
        if self.drift:
            return np.array(
                [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]] if self.use_acc else
                [[1, 0, 1], [0, 1, 0]]
            )
        else:
            return np.eye(self.n_dim_obs)

    def __filter_update_pos(self, T, position, vel, acc):
        measure_x = np.array([position[0], vel[0], acc[0]]) if self.use_acc else np.array([position[0], vel[0]])
        measure_y = np.array([position[1], vel[1], acc[1]]) if self.use_acc else np.array([position[1], vel[1]])
        matrix = self.get_transition_matrix(T)
        self.curr_mean_x, self.curr_cov_x = self.filter.filter_update(
            self.curr_mean_x, self.curr_cov_x, measure_x, transition_matrix=matrix)
        self.curr_mean_y, self.curr_cov_y = self.filter.filter_update(
            self.curr_mean_y, self.curr_cov_y, measure_y, transition_matrix=matrix)

    def __filter_update_disp(self, T, position, vel, acc):
        displacement = position[:2] - self.last_pos_obs
        self.last_pos_obs = position[:2]
        measure_x = np.array([displacement[0], vel[0], acc[0]]) if self.use_acc else np.array([displacement[0], vel[0]])
        measure_y = np.array([displacement[1], vel[1], acc[1]]) if self.use_acc else np.array([displacement[1], vel[1]])
        matrix = self.get_transition_matrix(T)
        self.curr_mean_x, self.curr_cov_x = self.filter.filter_update(
            self.curr_mean_x, self.curr_cov_x, measure_x, transition_matrix=matrix)
        self.curr_mean_y, self.curr_cov_y = self.filter.filter_update(
            self.curr_mean_y, self.curr_cov_y, measure_y, transition_matrix=matrix)

    def __call__(self, T, position, vel, acc):
        if self.pred_displ:
            self.__filter_update_disp(T, position, vel, acc)
            new_pos = np.array([self.last_pos_pred[0] + self.curr_mean_x[0], self.last_pos_pred[1] + self.curr_mean_y[0], position[2]])
            self.last_pos_pred = new_pos[:2]
        else:
            self.__filter_update_pos(T, position, vel, acc)
            new_pos = np.array([self.curr_mean_x[0], self.curr_mean_y[0], position[2]])
        new_vel = np.array([self.curr_mean_x[1], self.curr_mean_y[1], vel[2]])
        if self.use_acc:
            new_acc = np.array([self.curr_mean_x[2], self.curr_mean_y[2], acc[2]])
            return new_pos, new_vel, new_acc
        else:
            return new_pos, new_vel, acc # TODO: model angle
        
    def loglikelihood(self, pos, vel, acc):
        """Compute log likelihood of a state given the current filter state"""
        state_x = np.array([pos[0], vel[0], acc[0]]) if self.use_acc else np.array([pos[0], vel[0]])
        state_y = np.array([pos[1], vel[1], acc[1]]) if self.use_acc else np.array([pos[1], vel[1]])
        logprob_x = multivariate_normal.logpdf(state_x, self.curr_mean_x, self.curr_cov_x)
        logprob_y = multivariate_normal.logpdf(state_y, self.curr_mean_y, self.curr_cov_y)
        return logprob_x + logprob_y

class KalmanFilterObs(KalmanFilter):
    def __init__(self, period, pred_cov, obs_cov, init_cov, use_acc=False, pred_displ=False) -> None:
        super().__init__(pred_cov, obs_cov, init_cov, use_acc, pred_displ)
        self.period = period

    def __call__(self, obs):
        position = obs[AbsEnv.POSITION]
        angle = obs[AbsEnv.ANGLE]
        vel = obs[AbsEnv.VELOCITY]
        acc = obs[AbsEnv.ACCELERATION]
        newobs = copy.deepcopy(obs)
        new_pos, new_vel, new_acc = super()(self.period, np.concatenate([position, angle]), vel, acc)
        if self.pred_displ:
            newobs[AbsEnv.POSITION] = position + np.array([self.curr_mean_x[0], self.curr_mean_y[0]])
        else:
            newobs[AbsEnv.POSITION] = np.array([self.curr_mean_x[0], self.curr_mean_y[0]])
        newobs[AbsEnv.VELOCITY] = np.array([self.curr_mean_x[1], self.curr_mean_y[1]])
        if self.use_acc:
            newobs[AbsEnv.ACCELERATION] = np.array([self.curr_mean_x[2], self.curr_mean_y[2], acc[2]])
        return newobs 
