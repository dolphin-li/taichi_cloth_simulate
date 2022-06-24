import taichi as ti
import numpy as np
from ti_cloth_mesh import ClothMesh

@ti.data_oriented
class PbdSolver:
    def __init__(self, cloth_mesh, body_mesh, sim_param):
        self.cloth = cloth_mesh
        self.body = body_mesh
        self.sim_param = sim_param
        

    def update(self, dt = 0.033):
        pass