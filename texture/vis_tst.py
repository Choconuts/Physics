import imgui
import numpy as np
import trimesh

from texture_model import *
from interface import visualize_field, ui, Inputable


class Opt(Inputable):

    def __init__(self):
        self.p = (0.4, 0.4, 0.4)
        self.max_depth = 5
        self.min_depth = 2
        self.add = False

    def gui(self, label, *args, **kwargs):
        self.changed, self.max_depth = imgui.slider_int("max", self.max_depth, 1, 10)
        self.changed, self.min_depth = imgui.slider_int("min", self.min_depth, 1, self.max_depth)
        self.changed, self.p = imgui.input_float3('p', *self.p)
        self.changed, self.add = imgui.checkbox("add", self.add)

opt = Opt()


class Scene:

    def __init__(self):
        self.mesh = trimesh.load_mesh("./dev/lego_uv3.obj")
        self.vp = cv2.imread("vp.png") / 255
        self.vn = cv2.imread("vn.png") / 255
        self.ntm = NeuSTextureModel("cache", [-1.5, 1.5])

    def vis_points(self):
        pnts, norms = self.ntm.load_vertices_and_normals()
        visualize_field(pnts.reshape([-1, 3]), scalars=norms.reshape([-1, 3]))

    def vis_mesh(self):
        pnts = np.array(self.mesh.vertices)
        visualize_field(pnts.reshape([-1, 3]), scalars=pnts.reshape([-1, 3]))

    @ui(opt)
    def show_points(self):
        verts = np.array(self.mesh.vertices)
        grads = np.array(self.mesh.vertex_normals)
        pnts, norms = self.ntm.load_vertices_and_normals()
        while True:
            if opt.add:
                visualize_field(verts.reshape([-1, 3]), scalars=grads.reshape([-1, 3]))
            else:
                visualize_field(pnts.reshape([-1, 3]), scalars=norms.reshape([-1, 3]))
            yield


if __name__ == '__main__':
    scene = Scene()
    scene.show_points()
