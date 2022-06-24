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
        data_indices_coo = self._calc_indices_coo(self.tris.to_numpy())
        self.n_indices_coo = len(data_indices_coo)
        self.indices_coo = ti.Vector.field(2, ti.i32, self.n_indices_coo)
        self.indices_coo.from_numpy(np.array(data_indices_coo, dtype = np.int32))
        self.indices_csr_ptr = ti.field(ti.i32, self.n_verts + 1)
        self._compute_indices_csr()
        self.indices_diag = ti.field(ti.i32, self.n_verts)
        self._compute_indices_diag()
        self._tmp_count_e_of_t = ti.field(ti.i32, self.n_indices_coo)
        self.indices_tri_of_edge = ti.field(ti.i32, (self.n_indices_coo, 2))
        self._compute_indices_tri_of_edge()

    # compute via python, since taichi does not support reduction ops
    def _calc_indices_coo(self, data_triangles):
        data_indices_coo = []
        for i in range(len(data_triangles)):
            t = data_triangles[i]
            data_indices_coo.append((t[0], t[0]))
            data_indices_coo.append((t[0], t[1]))
            data_indices_coo.append((t[0], t[2]))
            data_indices_coo.append((t[1], t[0]))
            data_indices_coo.append((t[1], t[1]))
            data_indices_coo.append((t[1], t[2]))
            data_indices_coo.append((t[2], t[0]))
            data_indices_coo.append((t[2], t[1]))
            data_indices_coo.append((t[2], t[2]))
        data_indices_coo.sort()
        data_indices_coo = self._unique(data_indices_coo)
        return data_indices_coo

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
            
    @ti.kernel
    def _compute_indices_csr(self):
        for i in self.indices_csr_ptr:
            self.indices_csr_ptr[i] = 0
        for i in self.indices_coo:
            if i > 0:
                e_prev = self.indices_coo[i-1]
                e = self.indices_coo[i]
                if e[0] != e_prev[0]:
                    self.indices_csr_ptr[e[0]] = i
        self.indices_csr_ptr[self.n_verts] = self.n_indices_coo
        
    @ti.kernel
    def _compute_indices_diag(self):
        for i in self.indices_diag:
            self.indices_diag[i] = 0
        for i in self.indices_coo:
            e = self.indices_coo[i]
            if e[0] == e[1]:
                self.indices_diag[e[0]] = i

    @ti.kernel
    def _compute_indices_tri_of_edge(self):
        for i in self.indices_coo:
            self.indices_tri_of_edge[i,0] = -1
            self.indices_tri_of_edge[i,1] = -1
            self._tmp_count_e_of_t[i] = 0
        for i in self.tris:
            t = self.tris[i]
            index = self._find_coo_index(t[0], t[1])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self._find_coo_index(t[1], t[0])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self._find_coo_index(t[0], t[2])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self._find_coo_index(t[2], t[0])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self._find_coo_index(t[1], t[2])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i
            index = self._find_coo_index(t[2], t[1])
            if index >= 0:
                self.indices_tri_of_edge[index, ti.atomic_add(self._tmp_count_e_of_t[index], 1)] = i

    @ti.func
    def _find_coo_index(self, row, col):
        begin = self.indices_csr_ptr[row]
        end = self.indices_csr_ptr[row+1]
        index = -1
        for pos in range(begin, end):
            if self.indices_coo[pos][1] == col:
                index = pos
        return index
