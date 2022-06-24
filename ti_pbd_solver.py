import taichi as ti
import numpy as np
from ti_cloth_mesh import ClothMesh

@ti.data_oriented
class PbdSolver:
    def __init__(self, cloth_mesh, body_mesh, sim_param):
        self.cloth = cloth_mesh
        self.body = body_mesh
        self.sim_param = sim_param

        # build topology
        data_indices_coo = self._calc_indices_coo(self.cloth.tris.to_numpy(), self.cloth.edges.to_numpy(), self.cloth.indices_tri_of_edge.to_numpy())
        self.n_indices_coo = len(data_indices_coo)
        self.indices_coo = ti.Vector.field(2, ti.i32, self.n_indices_coo)
        self.indices_coo.from_numpy(np.array(data_indices_coo, dtype = np.int32))
        self.indices_csr_ptr = ti.field(ti.i32, self.cloth.n_verts + 1)
        self._compute_indices_csr()
        self.indices_diag = ti.field(ti.i32, self.cloth.n_verts)
        self._compute_indices_diag()

        # fixed verts
        self.verts_is_fixed = ti.field(ti.i32, self.cloth.n_verts)
        self.verts_is_fixed.fill(0)
        self.verts_is_fixed[0] = 1 # debug, fix one vertex
        
        # build numeric
        self.edges_length_rest = ti.field(ti.f32, self.n_indices_coo)
        self.mat_A = ti.field(ti.f32, self.n_indices_coo)
        self.vec_B = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
        self.vec_X = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
        self._compute_A()

    # compute via python, since taichi does not support reduction ops
    def _calc_indices_coo(self, data_triangles, data_edges, data_t_of_e):
        data_indices_coo = []
        for i in range(len(data_edges)):
    		# add original edges
            e = data_edges[i]
            data_indices_coo.append((e[0], e[1]))
            data_indices_coo.append((e[1], e[0]))
			# add bending edges
            i_t0 = data_t_of_e[i][0]
            i_t1 = data_t_of_e[i][1]
            if i_t0 >= 0 and i_t1 >= 0:
                t0 = data_triangles[i_t0]
                t1 = data_triangles[i_t1]
                v2 = t0[0] + t0[1] + t0[2] - e[0] - e[1]
                v3 = t1[0] + t1[1] + t1[2] - e[0] - e[1]
                data_indices_coo.append((v2, v3))
                data_indices_coo.append((v3, v2))
        for i in range(self.cloth.n_verts):
            data_indices_coo.append((i, i))
        data_indices_coo.sort()
        data_indices_coo = self.cloth.unique(data_indices_coo)
        return data_indices_coo

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
        self.indices_csr_ptr[self.cloth.n_verts] = self.n_indices_coo
        
    @ti.kernel
    def _compute_indices_diag(self):
        for i in self.indices_diag:
            self.indices_diag[i] = 0
        for i in self.indices_coo:
            e = self.indices_coo[i]
            if e[0] == e[1]:
                self.indices_diag[e[0]] = i

    @ti.func
    def find_coo_index(self, row, col):
        begin = self.indices_csr_ptr[row]
        end = self.indices_csr_ptr[row+1]
        index = -1
        for pos in range(begin, end):
            if self.indices_coo[pos][1] == col:
                index = pos
        return index
 
    @ti.kernel
    def _compute_A(self):
        # sim params
        spring_k = self.sim_param.spring_stiffness
        bending_k = self.sim_param.bending_stiffness
        dt = self.sim_param.dt
        fix_k = self.sim_param.fix_stiffness
        # reset A
        for i in self.mat_A:
            self.mat_A[i] = 0.0
        for i in self.edges_length_rest:
            self.edges_length_rest[i] = -1.0
        # init diag
        for i in self.indices_diag:
            i_v = self.indices_diag[i]
            fix_w = 0.0
            if self.verts_is_fixed[i_v]: fix_w = fix_k
            self.mat_A[i_v] = (self.cloth.verts_mass[i_v] + fix_w) / (dt * dt)
        # init A
        for i_e in self.cloth.edges:
            i_v = [0, 0, 0, 0]
            i_v[0] = self.cloth.edges[i_e][0]
            i_v[1] = self.cloth.edges[i_e][1]
            index_e01 = self.find_coo_index(i_v[0], i_v[1])
            index_e10 = self.find_coo_index(i_v[1], i_v[0])
            # edge length
            V0 = self.cloth.verts[i_v[0]]
            V1 = self.cloth.verts[i_v[1]]
            e_len = (V0 - V1).norm()
            self.edges_length_rest[index_e01] = e_len
            self.edges_length_rest[index_e10] = e_len
            # spring strain stiffness
            self.mat_A[index_e01] -= spring_k
            self.mat_A[index_e10] -= spring_k
            self.mat_A[self.indices_diag[i_v[0]]] += spring_k
            self.mat_A[self.indices_diag[i_v[1]]] += spring_k
            # bending stiffness
            i_t0 = self.cloth.indices_tri_of_edge[i_e, 0]
            i_t1 = self.cloth.indices_tri_of_edge[i_e, 1]
            if i_t0 >= 0 and i_t1 >= 0:
                t0 = self.cloth.tris[i_t0]
                t1 = self.cloth.tris[i_t1]
                i_v[2] = t0[0] + t0[1] + t0[2] - i_v[0] - i_v[1]
                i_v[3] = t1[0] + t1[1] + t1[2] - i_v[0] - i_v[1]
                V2 = self.cloth.verts[i_v[2]]
                V3 = self.cloth.verts[i_v[3]]
                c01 = self.cotangent(V0, V1, V2)
                c02 = self.cotangent(V0, V1, V3)
                c03 = self.cotangent(V1, V0, V2)
                c04 = self.cotangent(V1, V0, V3)
                area0 = self.area(V0, V1, V2)
                area1 = self.area(V1, V0, V3)
                weight = 1.0 / (area0 + area1)
                k = [0.0, 0.0, 0.0, 0.0]
                k[0]= c03+c04
                k[1]= c01+c02
                k[2]=-c01-c03
                k[3]=-c02-c04
                for r in ti.static(range(4)):
                    for c in ti.static(range(4)):
                        if r == c: self.mat_A[self.indices_diag[i_v[r]]] += k[r] * k[c] * bending_k * weight
                        else: self.mat_A[self.find_coo_index(i_v[r], i_v[c])] += k[r] * k[c] * bending_k * weight

    @ti.func
    def cotangent(self, x0, x1, x2):
        x10 = x1 - x0
        x20 = x2 - x0
        dot = x10.dot(x20)
        return dot / ti.sqrt(x10.dot(x10)*x20.dot(x20) - dot * dot)

    @ti.func
    def area(self, x0, x1, x2):
        x10 = x1 - x0
        x20 = x2 - x0
        normal = x10.cross(x20)
        return ti.sqrt(normal.dot(normal))  

    def update(self, dt = 0.033):
        pass