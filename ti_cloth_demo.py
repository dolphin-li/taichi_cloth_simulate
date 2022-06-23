import taichi as ti
import numpy as np
import pywavefront as pyw
from ti_base_mesh import BaseMesh
from arcball import ArcBall

@ti.data_oriented
class Cloth(BaseMesh):
    pass

@ti.data_oriented
class Body(BaseMesh):
    pass

class UI:
    def __init__(self):
        # load mesh
        body_obj = pyw.Wavefront('data/body.obj', collect_faces=True)
        cloth_obj = pyw.Wavefront('data/skirt.obj', collect_faces=True)

        # init ti
        self.cloth = Cloth(cloth_obj)
        self.body = Body(body_obj)

        # create window
        self.window = ti.ui.Window('ti_cloth', (640, 480))

        # other flags and data
        self.mouth_left_pressed = False
        self.mouth_right_pressed = False
        self.camera_R = np.eye(3, dtype = np.float32)
        self.camera_t = np.array([0, 0, 1])
        self.camera_up = np.array([0, 1, 0])
        self.mouse_pt = (0.0, 0.0)
        self.arcball = ArcBall()
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
        cam_u = np.dot(self.camera_R, self.camera_up)
        cam_p = np.dot(self.camera_R, self.camera_t)
        camera.position(cam_p[0], cam_p[1], cam_p[2])
        camera.lookat(0, 0, 0)
        camera.up(cam_u[0], cam_u[1], cam_u[2])
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
        self.arcball.click(pt)
        print('left mouse press event: ', pt)

    def left_mouse_drag_event(self, pt):
        self.camera_R, self.camera_t = self.arcball.drag(pt, self.camera_R, self.camera_t)

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
    