import taichi as ti
import numpy as np
from ti_base_mesh import BaseMesh

@ti.data_oriented
class ClothMesh(BaseMesh):
    def __init__(self, mesh_obj):
        super().__init__(mesh_obj)
        self.verts_mass = ti.field(ti.f32, self.n_verts)
        self.edges_length_rest = ti.field(ti.f32, self.n_edges)
        self.compute_verts_mass()
        self.compute_edge_length_rest()

    @ti.kernel
    def compute_verts_mass(self):
        for i in self.verts:
            self.verts_mass[i] = 0.0
        for i in self.tris:
            tri = self.tris[i]
            a = self.verts[tri[0]]
            b = self.verts[tri[1]]
            c = self.verts[tri[2]]
            area = (b-a).cross(c-a).norm()
            self.verts_mass[tri[0]] += area
            self.verts_mass[tri[1]] += area
            self.verts_mass[tri[2]] += area

    @ti.kernel
    def compute_edge_length_rest(self):
        for i in self.edges:
            e = self.edges[i]
            dist = (self.verts[e[0]] - self.verts[e[1]]).norm()
            self.edges_length_rest[i] = dist