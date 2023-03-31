import numpy as np

MAX_MOVE_POS = 5.
MIN_X_VEL = -3.5
MAX_X_VEL = 3.5
MIN_Y_VEL = -3.5
MAX_Y_VEL = 3.5
MIN_A_VEL = -600.
MAX_A_VEL = 600.
MIN_MOTOR_VEL = -8192.
MAX_MOTOR_VEL = 8191.
MIN_ARM_X = -np.inf
MIN_ARM_Y = -np.inf
MAX_ARM_X = -30.
MAX_ARM_Y = 30. # TODO: check arm coordinate range