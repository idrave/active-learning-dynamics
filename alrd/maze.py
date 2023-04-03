from shapely import Polygon,LineString, Point, dwithin, is_ccw
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation 

class Maze:
    def __init__(self, points, margin=0.22) -> None:
        # TODO: check if points are in clockwise order
        self.shape = Polygon(points)
        assert self.shape.is_valid, "Maze shape is not valid polygon"
        assert not is_ccw(self.shape.exterior), "Maze vertices must be supplied in clockwise order"
        self.margin = margin
        self.__normals = []
        for p, q in self.get_sides():
            normal = np.array([p[1]-q[1], q[0]-p[0]]) # (counterclockwise) normal to the line segment
            normal = normal / np.linalg.norm(normal)
            self.__normals.append(normal)

    def get_sides(self):
        p = None
        for q in self.shape.exterior.coords:
            if p is not None:
                yield (p,q)
            p = q
    
    @property
    def normals(self):
        return deepcopy(self.__normals)

    def is_inside(self, position):
        if not isinstance(position, Point):
            position = Point(position)
        return self.shape.contains(position) and not self.shape.exterior.dwithin(position, self.margin)
    
    def valid_move(self, position, angle, velocity):
        """
        Returns True if the robot is inside the maze with a certain margin, or if the velocity points away from the maze otherwise.
        :param position: robot position (x,y)
        :param angle: robot angle (degrees)
        :param velocity: robot velocity (vx,vy). vx is forward and vy is lateral movement
        """
        if not isinstance(position, Point):
            position = Point(position)
        velocity = np.array(velocity)
        velocity = Rotation.from_euler('z', angle).as_matrix()[:2, :2] @ velocity
        speed = np.linalg.norm(velocity)
        if speed < 1e-5:
            return True
        direction = velocity / speed
        if self.is_inside(position):
            return True
        else:
            for side, normal in zip(self.get_sides(), self.__normals):
                if LineString(side).dwithin(position, self.margin) and np.dot(normal, direction) > -1.e-4:
                    return False
            return True

    def clamp_direction(self, position, direction):
        if not isinstance(position, Point):
            position = Point(position)
        if self.is_inside(position):
            return direction
        else:
            for side, normal in zip(self.get_sides(), self.__normals):
                if LineString(side).dwithin(position, self.margin) and np.dot(normal, direction) > 0.0:
                    proj = np.dot(normal, direction)
                    direction -= proj * normal
            return direction