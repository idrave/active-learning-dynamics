from shapely import Polygon,LineString, Point, dwithin, is_ccw
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation 
from alrd.utils.utils import rotate_2d_vector

def create_maze(margin=0.3):
    p0 = np.array([-0.65, -0.77])
    #diffs = np.array([
    #    [0,1.6],
    #    [1.6,0],
    #    [0,1.6],
    #    [2.1,0],
    #    [0,-1.6],
    #    [-1.5,0],
    #    [0,-1.6],
    #    [1.5,0],
    #    [0,-1.6],
    #    [-2.1,0],
    #    [0,1.6],
    #])
    diffs = np.array([
        [0,1.6],
        [1.5,0],
        [0,1.6],
        [2.2,0],
        [0,-1.6],
        [-1.4,0],
        [0,-1.6],
        [1.4,0],
        [0,-1.6],
        [-2.2,0],
        [0,1.6],
    ])
    points = [p0]
    for d in diffs:
        points.append(points[-1]+d)
    coords = np.stack(points)
    return Maze(coords, margin=margin)

class Maze:
    def __init__(self, points, margin=0.22) -> None:
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
        :param velocity: robot velocity (vx,vy) in the global coordinate system
        """
        if not isinstance(position, Point):
            position = Point(position)
        velocity = np.array(velocity)
        #velocity = rotate_2d_vector(velocity, angle)
        speed = np.linalg.norm(velocity)
        if speed < 1e-5:
            return True
        direction = velocity / speed
        if self.is_inside(position):
            return True
        else:
            for side, normal in zip(self.get_sides(), self.__normals):
                if LineString(side).dwithin(position, self.margin) \
                        and np.dot(normal, direction) > 1.e-4:
                    return False
            return True

    def clamp_direction(self, position, angle, velocity):
        """
        Returns velocity "clamped" in such a way that it does not go through the maze walls given the current position.
        :param position: robot position (x,y)
        :param angle: robot angle (degrees)
        :param velocity: robot velocity (vx,vy) in the global coordinate system
        """
        if not isinstance(position, Point):
            position = Point(position)
        velocity = np.array(velocity)
        #velocity = rotate_2d_vector(velocity, angle)
        if not self.shape.contains(position):
            velocity = np.zeros_like(velocity)
            for side, normal in zip(self.get_sides(), self.__normals):
                if LineString(side).dwithin(position, self.margin):
                    velocity -= normal
            norm = np.linalg.norm(velocity)
            if norm > 1e-5:
                velocity = velocity / np.linalg.norm(velocity)
        else:
            for side, normal in zip(self.get_sides(), self.__normals):
                #if np.dot(np.array(position.coords[0]) - np.array(side[0]), normal) > -self.margin and np.dot(normal, velocity) > 0.0:
                if LineString(side).dwithin(position, self.margin) and np.dot(normal, velocity) > 0.0:
                    proj = np.dot(normal, velocity)
                    velocity -= proj * normal
        #return rotate_2d_vector(velocity, -angle)
        return velocity