import numpy as np
from pyquaternion import Quaternion

class ArcBall:
    def __init__(self):
        self.__st_vec = np.zeros(3, dtype=np.float32)
        self.__last_R = np.eye(3, dtype=np.float32)
        self.__center = np.zeros(3, dtype=np.float32)
    
    def click(self, pt):
        self.__st_vec = self.__sphere_map(pt)
        self.__last_R = np.eye(3, 3)

    def drag(self, pt, cam_R, cam_t):
        if np.linalg.det(self.__last_R) == 0.0:
            return
        q = Quaternion(axis=[1,0,0], angle=0.0)
        ed_vec = self.__sphere_map(pt)
        perp_vec = np.cross(self.__st_vec, ed_vec)
        if np.linalg.norm(perp_vec) > 1e-5:
            q = Quaternion(axis=perp_vec, angle=2.0 * np.arccos(np.dot(self.__st_vec, ed_vec)))
        R = np.dot(q.rotation_matrix, np.linalg.inv(self.__last_R))
        R = np.dot(R, cam_R)
        t = cam_t - np.dot(cam_R - R, self.__center)
        self.__last_R = q.rotation_matrix
        return R, t

    def set_center(self, c):
        self.center = c

    def get_center(self):
        return self.__center

    def __sphere_map(self, pt):
        x = -(pt[0] * 2.0 - 1.0)
        y = -(pt[1] * 2.0 - 1.0)
        len = x * x + y * y
        ret = np.zeros(3, dtype=np.float32)
        if len > 1.0:
            norm = 1.0 / np.sqrt(len)
            ret[0] = - x * norm
            ret[1] =  y * norm
            ret[2] = 0.0
        else:
            ret[0] = x
            ret[1] = y
            ret[2] = np.sqrt(1.0 - len)
        return ret
    