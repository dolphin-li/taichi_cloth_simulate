import taichi as ti
import numpy as np
from ti_cloth_mesh import ClothMesh

@ti.data_oriented
class PbdSolver:
    def __init__(self, cloth_mesh, body_mesh, sim_param):
        self.cloth = cloth_mesh
        self.body = body_mesh
        self.sim_param = sim_param
        self.gravity = ti.Vector.field(3, ti.f32, ())
        self.gravity[None] = sim_param.gravity

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
        self.vec_old_X = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
        self.vec_next_X = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
        self.vec_V = ti.Vector.field(3, ti.f32, self.cloth.n_verts)
        self._compute_A()

        # init value of X
        self.vec_X.copy_from(self.cloth.verts)
        self.vec_V.fill(0.0)

        # pcg variables
        self.vec_Ax = ti.Vector.field(3, ti.f32, self.cloth.n_verts)

    # compute via python, since taichi does not support reduction ops
    def _calc_indices_coo(self, data_triangles, data_edges, data_t_of_e):
        data_indices_coo = []
        for i_e in range(len(data_edges)):
    		# add original edges
            e = data_edges[i_e]
            data_indices_coo.append((e[0], e[1]))
            data_indices_coo.append((e[1], e[0]))
			# # add bending edges
            # i_t0 = data_t_of_e[i_e][0]
            # i_t1 = data_t_of_e[i_e][1]
            # if i_t0 >= 0 and i_t1 >= 0:
            #     t0 = data_triangles[i_t0]
            #     t1 = data_triangles[i_t1]
            #     v2 = t0[0] + t0[1] + t0[2] - e[0] - e[1]
            #     v3 = t1[0] + t1[1] + t1[2] - e[0] - e[1]
            #     data_indices_coo.append((v2, v3))
            #     data_indices_coo.append((v3, v2))
        for i_v in range(self.cloth.n_verts):
            data_indices_coo.append((i_v, i_v))
        data_indices_coo.sort()
        data_indices_coo = self.cloth.unique(data_indices_coo)
        return data_indices_coo

    @ti.kernel
    def _compute_indices_csr(self):
        for i_v in self.indices_csr_ptr:
            self.indices_csr_ptr[i_v] = 0
        for i_coo in self.indices_coo:
            if i_coo > 0:
                e_prev = self.indices_coo[i_coo-1]
                e = self.indices_coo[i_coo]
                if e[0] != e_prev[0]:
                    self.indices_csr_ptr[e[0]] = i_coo
        self.indices_csr_ptr[self.cloth.n_verts] = self.n_indices_coo
        
    @ti.kernel
    def _compute_indices_diag(self):
        for i_v in self.indices_diag:
            self.indices_diag[i_v] = self.find_coo_index(i_v, i_v)

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
        for i_coo in self.mat_A:
            self.mat_A[i_coo] = 0.0
        for i_coo in self.edges_length_rest:
            self.edges_length_rest[i_coo] = -1.0
        # init diag
        for i_v in self.indices_diag:
            i_coo = self.indices_diag[i_v]
            fix_w = 0.0
            if self.verts_is_fixed[i_v]: fix_w = fix_k
            self.mat_A[i_coo] = (self.cloth.verts_mass[i_v] + fix_w) / (dt * dt)
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
            # # bending stiffness
            # i_t0 = self.cloth.indices_tri_of_edge[i_e, 0]
            # i_t1 = self.cloth.indices_tri_of_edge[i_e, 1]
            # if i_t0 >= 0 and i_t1 >= 0:
            #     t0 = self.cloth.tris[i_t0]
            #     t1 = self.cloth.tris[i_t1]
            #     i_v[2] = t0[0] + t0[1] + t0[2] - i_v[0] - i_v[1]
            #     i_v[3] = t1[0] + t1[1] + t1[2] - i_v[0] - i_v[1]
            #     V2 = self.cloth.verts[i_v[2]]
            #     V3 = self.cloth.verts[i_v[3]]
            #     c01 = self.cotangent(V0, V1, V2)
            #     c02 = self.cotangent(V0, V1, V3)
            #     c03 = self.cotangent(V1, V0, V2)
            #     c04 = self.cotangent(V1, V0, V3)
            #     area0 = self.area(V0, V1, V2)
            #     area1 = self.area(V1, V0, V3)
            #     weight = 1.0 / (area0 + area1)
            #     k = [0.0, 0.0, 0.0, 0.0]
            #     k[0]= c03+c04
            #     k[1]= c01+c02
            #     k[2]=-c01-c03
            #     k[3]=-c02-c04
            #     for r in ti.static(range(4)):
            #         for c in ti.static(range(4)):
            #             if r == c: self.mat_A[self.indices_diag[i_v[r]]] += k[r] * k[c] * bending_k * weight
            #             else: self.mat_A[self.find_coo_index(i_v[r], i_v[c])] += k[r] * k[c] * bending_k * weight

    @ti.kernel
    def _update_B(self):
        dt = self.sim_param.dt
        fix_k = self.sim_param.fix_stiffness
        spring_k = self.sim_param.spring_stiffness
        g = self.gravity[None]
        for i_v in self.vec_B:
            fix_w = 0.0
            if self.verts_is_fixed[i_v]: fix_w = fix_k
            mass = self.cloth.verts_mass[i_v]
            self.vec_B[i_v] = (mass + fix_w) / (dt * dt) * self.vec_old_X[i_v] + mass * g
        for i_coo in self.indices_coo:
            rest_len = self.edges_length_rest[i_coo]
            if rest_len >= 0.0:
                e = self.indices_coo[i_coo]
                dif = self.vec_X[e[0]] - self.vec_X[e[1]]
                new_len = spring_k*rest_len/dif.norm();		
                self.vec_B[e[0]] += new_len * dif


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

    @ti.func
    def A_mult_x(self, x, Ax):
        for i in Ax:
            Ax[i] = (0.0, 0.0, 0.0)
        for i in self.indices_coo:
            e = self.indices_coo[i]
            Ax[e[0]] += self.mat_A[i] * x[e[1]]

    @ti.func
    def x_dot_y(self, x, y):
        sum = (0.0, 0.0, 0.0)
        for i in x:
            sum[0] += x[i][0] * y[i][0]
            sum[1] += x[i][1] * y[i][1]
            sum[2] += x[i][2] * y[i][2]
        return sum

    @ti.kernel
    def _apply_gravity(self):
        dt = self.sim_param.dt
        g = self.gravity[None]
        for i_v in self.vec_X:
            if not self.verts_is_fixed[i_v]:
                self.vec_V[i_v] += g * dt
                self.vec_X[i_v] += self.vec_V[i_v] * dt

    @ti.kernel 
    def _update_V(self):
        dt = self.sim_param.dt
        for i_v in self.vec_V:
            self.vec_V[i_v] = (self.vec_X[i_v] - self.vec_old_X[i_v]) / dt
    
    @ti.kernel
    def _pcg_one_iteration(self):
        pass

    @ti.kernel
    def _pcg_init(self):
        pass

    @ti.kernel
    def _jacobi_one_iteration(self):
        # x = B
        for i_v in self.vec_next_X:
            self.vec_next_X[i_v] = self.vec_B[i_v]
        # x -= (L + U)x
        for i_coo in self.mat_A:
            e = self.indices_coo[i_coo]
            if e[0] != e[1]:
                self.vec_next_X[e[0]] -= self.mat_A[i_coo] * self.vec_X[e[1]] 
        # x = D^(-1)x
        for i_v in self.vec_next_X:
            diag = self.mat_A[self.indices_diag[i_v]]
            self.vec_next_X[i_v] /= diag

    def _update_one_time_step(self):
        # air damping
        
        # explicitly apply gravity
        # self._apply_gravity()

        # pcg process
        self.vec_old_X.copy_from(self.vec_X)
        for iter in range(20):
            self._update_B()
            self._jacobi_one_iteration()
            self.vec_X.copy_from(self.vec_next_X)
        self._update_V()

    def update(self, dt = 0.033):
        num_outer_iter = round(dt / self.sim_param.dt)
        for outer_iter in range(num_outer_iter):
            self._update_one_time_step()