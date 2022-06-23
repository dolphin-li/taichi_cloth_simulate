import taichi as ti
import numpy as np
import pywavefront as pyw
import quaternion

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
        self.normal_weights = ti.field(ti.f32, self.n_verts)
        self.verts.from_numpy(data_verts)
        self.tris.from_numpy(data_triangles)
        self.vcolors.fill(1.0)
        self.update_normal()
    
    @ti.kernel
    def update_normal(self):
        for i in self.verts:
            self.normal_weights[i] = 0.0
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
            self.normal_weights[tri[0]] += area
            self.normal_weights[tri[1]] += area
            self.normal_weights[tri[2]] += area
        for i in self.verts:
            w = self.normal_weights[i]
            if w != 0.0:
                self.vnormals[i] /= w    

@ti.data_oriented
class Cloth(BaseMesh):
    pass

@ti.data_oriented
class Body(BaseMesh):
    pass

class ArcBall:
    def __init__(self):
        self.__st_vec = np.zeros(3, 1)
        self.__last_R = np.zeros(3, 3)
        self.__center = np.zeros(3, 1)

    def set_center(self, c):
        self.center = c

    def get_center(self):
        return self.__center
    
    def click(self, pt):
        self.__st_vec = self.__sphere_map(pt)
        self.__last_R = np.eye(3, 3)

    def drag(self, pt):
        if np.linalg.det(self.__last_R) == 0.0:
            return
        ed_vec = self.__sphere_map(pt)
        perp_vec = np.cross(self.__st_vec, ed_vec)
        if np.norm(perp_vec) < 1e-5:
            pass
        else:
            pass

    def __sphere_map(self, pt):
        x = pt[0] * 2.0 - 1.0
        y = pt[1] * 2.0 - 1.0
        len = x * x + y * y
        ret = np.zeros(3, 1)
        if len > 1.0:
            norm = 1.0 / np.sqrt(len)
            ret[0] = x * norm
            ret[1] = y * norm
            ret[2] = 0.0
        else:
            ret[0] = x * norm
            ret[1] = y * norm
            ret[2] = np.sqrt(1.0 - len)
        return ret


class UI:
    def __init__(self):
        # load mesh
        body_obj = pyw.Wavefront('body.obj', collect_faces=True)
        cloth_obj = pyw.Wavefront('skirt.obj', collect_faces=True)

        # init ti
        self.cloth = Cloth(cloth_obj)
        self.body = Body(body_obj)

        # create window
        self.window = ti.ui.Window('ti_cloth', (640, 480))

        # other flags and data
        self.mouth_left_pressed = False
        self.mouth_right_pressed = False
        self.mouse_pt = (0.0, 0.0)
        self.should_exit = False

    def render(self):
        while self.window.running:
            self.update_event()
            self.update_canvas()

            # flush render
            self.window.show()
            if self.should_exit: break

    def update_canvas(self):
        canvas = self.window.get_canvas()
        canvas.set_background_color((0,0,0))

        # set camera
        camera = ti.ui.make_camera()
        camera.projection_mode(ti.ui.ProjectionMode(0))
        camera.position(0, 0, 1.5)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)
        camera.fov(45)

        # config scene
        scene = ti.ui.Scene()
        scene.set_camera(camera)
        scene.point_light((10, 10, 10), (1, 1, 1))
        scene.ambient_light((0.1, 0.1, 0.1))

        # render mesh
        self.cloth.update_normal()
        scene.mesh(self.cloth.verts, self.cloth.tris, self.cloth.vnormals, (0.9,0.6,0.3))
        scene.mesh(self.body.verts, self.body.tris, self.body.vnormals, (1.0,1.0,1.0))

        # set scene
        canvas.scene(scene)

    def mouse_moved(self, lp, p):
        dx = lp[0] - p[0]
        dy = lp[1] - p[1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 1e-7:
            return True
        return False

    def update_event(self):
        # keyboard event processing
        if self.window.get_event(ti.ui.PRESS):
            if self.window.event.key in [ti.ui.ESCAPE]: 
                self.should_exit = True
        
        # mouse event processing
        last_p = self.mouse_pt;
        self.mouse_pt = self.window.get_cursor_pos()
        if self.window.is_pressed(ti.ui.LMB):
            if not self.mouth_left_pressed:
                self.left_mouse_press_event(self.mouse_pt)
                self.mouth_left_pressed = True
            else:
                if self.mouse_moved(last_p, self.mouse_pt):
                    self.left_mouse_drag_event(self.mouse_pt)
        else:
            if self.mouth_left_pressed:
                self.left_mouse_release_event(self.mouse_pt)
                self.mouth_left_pressed = False
        if self.window.is_pressed(ti.ui.RMB):
            if not self.mouth_right_pressed:
                self.right_mouse_press_event(self.mouse_pt)
                self.mouth_right_pressed = True
            else:
                if self.mouse_moved(last_p, self.mouse_pt):
                    self.right_mouse_drag_event(self.mouse_pt)
        else:
            if self.mouth_right_pressed:
                self.right_mouse_release_event(self.mouse_pt)
                self.mouth_right_pressed = False

    def left_mouse_press_event(self, pt):
        print('left mouse press event: ', pt)

    def left_mouse_drag_event(self, pt):
        print('left mouse drag event: ', pt)

    def left_mouse_release_event(self, pt):
        print('left mouse release event: ', pt)

    def right_mouse_press_event(self, pt):
        print('right mouse press event: ', pt)

    def right_mouse_drag_event(self, pt):
        print('right mouse drag event: ', pt)

    def right_mouse_release_event(self, pt):
        print('right mouse release event: ', pt)


if __name__ == "__main__":
    ti.init(arch=ti.gpu, kernel_profiler=True)

    ui = UI()
    ui.render()
    