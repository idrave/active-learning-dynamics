import numpy as np

MAX_MOVE_POS = 5.
MIN_X = -MAX_MOVE_POS
MAX_X = MAX_MOVE_POS
MIN_Y = -MAX_MOVE_POS
MAX_Y = MAX_MOVE_POS
MIN_X_VEL = -1
MAX_X_VEL = 1
MIN_Y_VEL = -1
MAX_Y_VEL = 1
MIN_A_VEL = -200.
MAX_A_VEL = 200.
MIN_MOTOR_VEL = -8192.
MAX_MOTOR_VEL = 8191.
MIN_ARM_X = -np.inf
MIN_ARM_Y = -np.inf
MAX_ARM_X = -30.
MAX_ARM_Y = 30. # TODO: check arm coordinate range