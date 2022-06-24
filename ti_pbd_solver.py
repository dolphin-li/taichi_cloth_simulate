import taichi as ti
import numpy as np
from ti_cloth_mesh import ClothMesh

@ti.data_oriented
class PbdSolver:
    def __init__(self, cloth_mesh, body_mesh, sim_param):
        self.cloth = cloth_mesh
        self.body = body_mesh
        self.sim_param = sim_param
        
    #     self.mat_A_diag = ti.field(ti.f32, self.cloth.n_verts)
    #     self.mat_A_other = ti.field(ti.f32, self.cloth.n_edges)
    #     self.vec_B = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
    #     self.vec_X = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
        
    # @ti.kernel
    # def compute_A(self):
    #     for i in self.mat_A_diag:
    #         self.mat_A_diag[i] = 0.0
    #     for i in self.cloth.edges:
    #         e = self.cloth.edges[i]
            
    #         # handle spring length
    #         self.mat_A_other[i] = -self.sim_param.spring_stiffness
    #         self.mat_A_diag[e[0]] += self.sim_param.spring_stiffness * 0.5
    #         self.mat_A_diag[e[1]] += self.sim_param.spring_stiffness * 0.5
        

    def update(self, dt = 0.033):
        pass