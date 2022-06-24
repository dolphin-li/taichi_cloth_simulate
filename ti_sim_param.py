import taichi as ti
import numpy as np
from ti_cloth_mesh import ClothMesh

@ti.data_oriented
class SimParam:
    def __init__(self, total_mass, spring_stiffness, fix_stiffness, bending_stiffness, dt):        
        self.total_mass = total_mass
        self.spring_stiffness = spring_stiffness
        self.bending_stiffness = bending_stiffness
        self.fix_stiffness = fix_stiffness
        self.dt = dt