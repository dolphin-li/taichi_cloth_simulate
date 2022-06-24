import taichi as ti
import numpy as np
import pywavefront as pyw

@ti.data_oriented
class BaseMesh:
    def __init__(self, mesh_obj):
        self.n_verts = len(mesh_obj.vertices)
        data_verts = np.array(mesh_obj.vertices, dtype=np.float32)
        self.n_tris = 0
        data_triangles = []
        for name in mesh_obj.meshes:
            mesh = mesh_obj.meshes[name]
            self.n_tris += len(mesh.faces)
            if not data_triangles:
                data_triangles = mesh.faces
            else:
                data_triangles.append(mesh.faces)
        data_triangles = np.array(data_triangles, dtype = np.int32)
        self.verts = ti.Vector.field(3, ti.f32, self.n_verts)
        self.tris = ti.Vector.field(3, ti.i32, self.n_tris)
        self.vnormals = ti.Vector.field(3, ti.f32, self.n_verts)
        self.vcolors = ti.Vector.field(3, ti.f32, self.n_verts)
        self._normal_weights = ti.field(ti.f32, self.n_verts)
        self.verts.from_numpy(data_verts)
        self.tris.from_numpy(data_triangles)
        self.vcolors.fill(1.0)
        self.update_normal()

    def _unique(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
    @ti.kernel
    def update_normal(self):
        for i in self.verts:
            self._normal_weights[i] = 0.0
            self.vnormals[i] = (0.0, 0.0, 0.0)
        for i in self.tris:
            tri = self.tris[i]
            a = self.verts[tri[0]]
            b = self.verts[tri[1]]
            c = self.verts[tri[2]]
            dir = (b-a).cross(c-a)
            area = dir.norm()
            self.vnormals[tri[0]] += dir
            self.vnormals[tri[1]] += dir
            self.vnormals[tri[2]] += dir
            self._normal_weights[tri[0]] += area
            self._normal_weights[tri[1]] += area
            self._normal_weights[tri[2]] += area
        for i in self.verts:
            w = self._normal_weights[i]
            if w != 0.0:
                self.vnormals[i] /= w

    