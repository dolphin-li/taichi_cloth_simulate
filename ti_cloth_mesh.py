import taichi as ti
import numpy as np
from ti_base_mesh import BaseMesh

@ti.data_oriented
class ClothMesh(BaseMesh):
    def __init__(self, mesh_obj, sim_param):
        super().__init__(mesh_obj)
        self.total_mass = sim_param.total_mass
        self.verts_mass = ti.field(ti.f32, self.n_verts)
        self._compute_verts_mass()
        data_edges = self._calc_edges(self.tris.to_numpy())
        self.n_edges = len(data_edges)
        self.edges = ti.Vector.field(2, ti.i32, self.n_edges)
        self.edges.from_numpy(np.array(data_edges, dtype = np.int32))
        self.edges_range_ptr = ti.Vector.field(2, ti.i32, self.n_verts)
        self._calc_edges_range()
        self._tmp_count_e_of_t = ti.field(ti.i32, self.n_edges)
        self.indices_tri_of_edge = ti.field(ti.i32, (self.n_edges, 2))
        self._compute_indices_tri_of_edge()

    # compute via python, since taichi does not support reduction ops
    def _calc_edges(self, data_triangles):
        data_edges = []
        for i in range(len(data_triangles)):
            t = data_triangles[i]
            if(t[0] < t[1]): data_edges.append((t[0], t[1]))
            else:  data_edges.append((t[1], t[0]))
            if(t[0] < t[2]): data_edges.append((t[0], t[2]))
            else:  data_edges.append((t[2], t[0]))
            if(t[1] < t[2]): data_edges.append((t[1], t[2]))
            else:  data_edges.append((t[2], t[1]))
        data_edges.sort()
        data_edges = self.unique(data_edges)
        return data_edges

    @ti.kernel
    def _calc_edges_range(self):
        for i in self.edges_range_ptr:
            self.edges_range_ptr[i] = (0, 0)
        for i in self.edges:
            if i > 0:
                e_prev = self.edges[i-1]
                e = self.edges[i]
                if e[0] != e_prev[0]:
                    self.edges_range_ptr[e_prev[0]][1] = i
                    self.edges_range_ptr[e[0]][0] = i
                if i == self.n_edges - 1:
                    self.edges_range_ptr[e[0]][1] = self.n_edges

    @ti.func
    def find_edge_index(self, row, col):
        x = ti.min(row, col)
        y = ti.max(row, col)
        begin = self.edges_range_ptr[x][0]
        end = self.edges_range_ptr[x][1]
        index = -1
        for pos in range(begin, end):
            if self.edges[pos][1] == y:
                index = pos
        return index

    @ti.kernel
    def _compute_indices_tri_of_edge(self):
        for i in self.edges:
            self.indices_tri_of_edge[i,0] = -1
            self.indices_tri_of_edge[i,1] = -1
            self._tmp_count_e_of_t[i] = 0
        for i in self.tris:
            t = self.tris[i]
            index = self.find_edge_index(t[0], t[1])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self.find_edge_index(t[0], t[2])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self.find_edge_index(t[1], t[2])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i

    @ti.kernel
    def _compute_verts_mass(self):
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
        sum_mass = 0.0
        for i in self.verts_mass:
            sum_mass += self.verts_mass[i]
        mass_scale = self.total_mass / sum_mass
        for i in self.verts:
            self.verts_mass[i] *= mass_scale