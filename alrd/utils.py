from datetime import datetime
from scipy.spatial.transform import Rotation
def get_timestamp_str():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def rotate_2d_vector(vector, angle):
    """
    Rotates a 2d vector by angle degrees
    :param vector: 2d vector
    :param angle: angle in degrees
    :return: rotated vector
    """
    return Rotation.from_euler('z', angle, degrees=True).as_matrix()[:2, :2] @ vector