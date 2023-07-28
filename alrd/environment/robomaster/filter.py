import pykalman
import numpy as np
from alrd.environment.robomaster.env import BaseRobomasterEnv
from alrd.utils.utils import rotate_2d_vector
import copy
import logging
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

def assert_param_shape(name, param, shape):
    assert param.shape == shape, f"Expected shape {shape} for {name}, got {param.shape}"

class KinematicKalmanFilter:
    def __init__(self, init_cov, pred_cov, obs_cov) -> None:
        """
        Kalman filter for kinematic models.
        :param init_cov: Initial covariance matrix
        :param pred_cov: Covariance matrix for the prediction step
        :param obs_cov: Covariance matrix for the observation step
        """
        self.n_dim_state = init_cov.shape[0]
        self.n_dim_obs = obs_cov.shape[0]
        assert_param_shape("init_cov", init_cov, (self.n_dim_state, self.n_dim_state))
        assert_param_shape("pred_cov", pred_cov, (self.n_dim_state, self.n_dim_state))
        assert_param_shape("obs_cov", obs_cov, (self.n_dim_obs, self.n_dim_obs))
        inv_fact = [1]
        for i in range(2, self.n_dim_state+1):
            inv_fact.append(inv_fact[-1] / i)
        self.__taylor_coeff = np.array(inv_fact)
        self.filter = pykalman.KalmanFilter(
            transition_matrices=None,
            observation_matrices=self.get_observation_matrix(),
            transition_covariance=pred_cov,
            observation_covariance=obs_cov,
            initial_state_mean=np.zeros(self.n_dim_state),
            initial_state_covariance=init_cov)
        self.curr_mean = self.filter.initial_state_mean
        self.curr_cov = init_cov

    def get_transition_matrix(self, T):
        taylor_coeff = np.array(self.__taylor_coeff)
        powers = [1]
        for i in range(1, self.n_dim_state):
            powers.append(powers[-1] * T)
        taylor_coeff *= np.array(powers)
        matrix = np.zeros((self.n_dim_state, self.n_dim_state))
        for i in range(self.n_dim_state):
            matrix[i, i:] = taylor_coeff[:self.n_dim_state - i]
        return matrix
    
    def get_observation_matrix(self):
        matrix = np.eye(self.n_dim_obs, self.n_dim_state)
        return matrix
    
    def __call__(self, T, obs):
        matrix = self.get_transition_matrix(T)
        self.curr_mean, self.curr_cov = self.filter.filter_update(
            self.curr_mean, self.curr_cov, obs, transition_matrix=matrix)
        return self.curr_mean, self.curr_cov
        
    def loglikelihood(self, obs):
        """Compute log likelihood of a state given the current filter state"""
        logprob = multivariate_normal.logpdf(obs, self.curr_mean, self.curr_cov)
        return logprob


class DriftFilter(KinematicKalmanFilter):
    def __init__(self, init_cov, pred_cov, obs_cov) -> None:
        """Assumes last state dimension is drift"""
        super().__init__(init_cov, pred_cov, obs_cov)

    def get_transition_matrix(self, T):
        matrix = super().get_transition_matrix(T)
        matrix[:, -1] = 0
        matrix[-1, -1] = 1
        return matrix
    
    def __call__(self, T, obs):
        mean, cov = super().__call__(T, obs)
        return mean[:-1], cov[:-1,:-1] # drop the drift

    def loglikelihood(self, obs):
        """Compute log likelihood of a state given the current filter state"""
        logprob = multivariate_normal.logpdf(obs, self.curr_mean[:,-1], self.curr_cov[:-1,:-1])
        return logprob
    

class DisplacementFilter(KinematicKalmanFilter):
    def __init__(self, init_cov, pred_cov, obs_cov) -> None:
        # the first state dimension is displacement
        super().__init__(init_cov, pred_cov, obs_cov)
        self.__last_obs = None
        self.__last_pred = None

    def get_transition_matrix(self, T):
        matrix = super().get_transition_matrix(T)
        matrix[0,0] = 0
        return matrix
    
    def __call__(self, T, obs):
        if self.__last_obs is not None:
            displ = obs[0] - self.__last_obs
        else:
            displ = 0
        measure = np.array(obs)
        measure[0] = displ
        matrix = self.get_transition_matrix(T)
        self.curr_mean, self.curr_cov = self.filter.filter_update(
            self.curr_mean, self.curr_cov, measure, transition_matrix=matrix)
        pred = np.array(self.curr_mean)
        if self.__last_pred is not None:
            pred[0] += self.__last_pred # else it will be 0
        self.__last_pred = pred[0]
        self.__last_obs = obs[0]
        return self.curr_mean, self.curr_cov
        
    def loglikelihood(self, obs):
        raise NotImplementedError("Not implemented for displacement filter")


class KalmanFilter:
    def __init__(self, pred_cov_x, obs_cov_x, init_cov_x, pred_cov_y, obs_cov_y, init_cov_y,
                 pred_cov_a, obs_cov_a, init_cov_a, drift_pos=False, drift_angle=False) -> None:
        if not drift_pos:
            self.filter_x = KinematicKalmanFilter(init_cov_x, pred_cov_x, obs_cov_x)
            self.filter_y = KinematicKalmanFilter(init_cov_y, pred_cov_y, obs_cov_y)
        else:
            self.filter_x = DriftFilter(init_cov_x, pred_cov_x, obs_cov_x)
            self.filter_y = DriftFilter(init_cov_y, pred_cov_y, obs_cov_y)
        if not drift_angle:
            self.filter_a = KinematicKalmanFilter(init_cov_a, pred_cov_a, obs_cov_a)
        else:
            self.filter_a = DriftFilter(init_cov_a, pred_cov_a, obs_cov_a)

    def __call__(self, T, obs_x, obs_y, obs_a):
        pred_x, _ = self.filter_x(T, obs_x)
        pred_y, _ = self.filter_y(T, obs_y)
        pred_a, _ = self.filter_a(T, obs_a)
        return pred_x, pred_y, pred_a
        
    def loglikelihood(self, obs_x, obs_y, obs_a):
        """Compute log likelihood of a state given the current filter state"""
        logprob_x = self.filter_x.loglikelihood(obs_x)
        logprob_y = self.filter_y.loglikelihood(obs_y)
        logprob_a = self.filter_a.loglikelihood(obs_a)
        return logprob_x + logprob_y + logprob_a

class KalmanFilterObs:
    def __init__(self, filter: KalmanFilter) -> None:
        self.filter = filter
        self.order_x = max(3, filter.filter_x.n_dim_obs)
        self.order_y = max(3, filter.filter_y.n_dim_obs)
        self.order_a = max(2, filter.filter_a.n_dim_obs)

    def __call__(self, obs):
        position = obs[BaseRobomasterEnv.POSITION]
        vel = obs[BaseRobomasterEnv.VELOCITY]
        angle = obs[BaseRobomasterEnv.ANGLE]
        angular_vel = obs[BaseRobomasterEnv.ANGULAR_V]
        acc = obs[BaseRobomasterEnv.ACCELERATION]
        newobs = copy.deepcopy(obs)
        obs_x = np.array([position[0], vel[0], acc[0]])
        obs_y = np.array([position[1], vel[1], acc[1]])
        obs_a = np.array([angle, angular_vel])
        pred_x = np.array(obs_x)
        pred_y = np.array(obs_y)
        pred_a = np.array(obs_a)
        pred_x[:self.order_x], pred_y[:self.order_y], pred_a[:self.order_a] = \
            self.filter(obs_x[:self.order_x], obs_y[:self.order_y], obs_a[:self.order_a])
        newobs[BaseRobomasterEnv.POSITION] = np.array([pred_x[0], pred_y[0]])
        newobs[BaseRobomasterEnv.VELOCITY] = np.array([pred_x[1], pred_y[1]])
        newobs[BaseRobomasterEnv.ACCELERATION] = np.array([pred_x[2], pred_y[2]])
        newobs[BaseRobomasterEnv.ANGLE] = pred_a[0]
        newobs[BaseRobomasterEnv.ANGULAR_V] = pred_a[1]
        return newobs 
