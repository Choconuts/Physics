import glob
import os.path
import time
import numpy as np
import torch
import trimesh
import xatlas
from trimesh.visual import material
from PIL import Image
import cv2
from torch import nn
from cluster.match import ImplicitNetworkMy
from texture_maker import render_texture, gen_uv_map

import pyglet

# pyglet.gl.glEnable(pyglet.gl.GL_FRAMEBUFFER_SRGB)


def erode_image(img_path, k=3):
    img = cv2.imread(img_path)
    kernel = np.ones((k, k), np.uint8)
    img_1 = cv2.erode(img, kernel, iterations=1)
    img_2 = cv2.erode(img, kernel, iterations=2)
    img[img > 254] = img_1[img > 254]
    img[img > 254] = img_2[img > 254]
    # cv2.imshow("tmp", img)
    # cv2.waitKey(0)
    cv2.imwrite(img_path, img)


class NeuSTextureRenderer:

    def __init__(self):
        self.neus = ImplicitNetworkMy()
        self.neus.cuda()
        self.gamma = 2.2

    def int_surf(self, x, n, steps=16):
        t = torch.linspace(-1, 1, steps, device=x.device) * 0.01
        x = x[:, None, :].expand(-1, steps, -1)
        n = n[:, None, :].expand(-1, steps, -1)
        x = x + n * t[None, :, None]
        n = n.reshape(-1, 3)
        f = self.neus.forward(x.reshape(-1, 3))
        c = self.neus.color(x.reshape(-1, 3), n, n, f[..., 1:])
        w = torch.exp(-t[:, None] ** 2 * 5e4)
        return (c.reshape(x.shape) * w).sum(-2) / w.sum(-2)

    def render_neus(self, x, n, chunk=8192):
        x = x * 0.5
        with torch.no_grad():
            res = []
            for i in range(0, x.shape[0], chunk):
                res.append(self.int_surf(x[i:i+chunk], n[i:i+chunk]))
        color = torch.cat(res, 0)
        return color ** (1 / self.gamma)

    def render_texture(self, mesh_path, out_path):

        def mat(mesh):
            vertices = torch.tensor(np.array(mesh.vertices)).cuda().float()
            normals = torch.tensor(np.array(mesh.vertex_normals)).cuda().float()
            colors = self.render_neus(vertices, normals)
            return trimesh.visual.color.ColorVisuals(mesh, vertex_colors=colors.cpu().numpy())

        render_texture(mesh_path, mat, out_path, erode=0)
        erode_image(out_path)

    def show_model(self, mesh_path, tex_path):
        img = Image.open(tex_path)
        mesh = trimesh.load(mesh_path)
        tex_visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, image=img)
        mesh.visual = tex_visual
        mesh.show()


def render_mesh_texture(mesh, material_fn, out_path, resolution=1024, uv=None):
    if uv is None:
        uv = np.array(mesh.visual.uv)
    color = material_fn(mesh)

    def get_render_texture(u, v):
        uvw = np.concatenate([u, v, np.zeros_like(uv[..., :1])], -1)
        mesh.vertices = uvw
        mesh.visual = color
        scene = trimesh.scene.scene.Scene(mesh)
        img = scene.save_image((resolution, resolution))
        with open(out_path, "bw") as fp:
            fp.write(img)
        return cv2.imread(out_path)

    front = get_render_texture(uv[..., :1], uv[..., 1:])
    back = get_render_texture(uv[..., 1:], uv[..., :1])
    back = back.transpose(1, 0, 2)
    back = np.flip(back, 0)
    back = np.flip(back, 1)
    combine = np.minimum(front, back)
    cv2.imwrite(out_path, combine)


class NeuSTextureModel(nn.Module):

    def __init__(self, mesh_path, bounding_box, resolution=1024):
        super(NeuSTextureModel, self).__init__()
        cache_dir = ".".join(os.path.basename(mesh_path).split(".")[:-1]) + ".cache"
        cache_dir = os.path.join(os.path.dirname(mesh_path), cache_dir)
        if os.path.exists(cache_dir):
            print("[Use Cache]", cache_dir)
        else:
            os.makedirs(cache_dir)
            print("[Make Cache]", cache_dir)
        self.cache_dir = cache_dir
        self.mesh = trimesh.load(mesh_path)
        if not hasattr(self.mesh.visual, "uv"):
            out_path = ".".join(mesh_path.split(".")[:-1]) + ".obj"
            assert out_path != mesh_path
            if not os.path.exists(out_path):
                print("[Parameterize] generate uv map")
                gen_uv_map(mesh_path, out_path)
            self.mesh = trimesh.load(out_path)
        self.uv = np.array(self.mesh.visual.uv)
        self.resolution = resolution
        box_min, box_max = bounding_box
        self.box_min = np.array(box_min)
        self.box_size = np.array(box_max) - self.box_min

        if not os.path.exists(self.get_cache_path("vertices.v0")):
            print("[Cache] generate vertices and normals")
            self.save_vertices_and_normals()
        if not os.path.exists(self.get_cache_path("mask")):
            print("[Cache] generate mask map")
            self.save_mask()

        self.multiplier = 1.5

    def save_mask(self):
        x = np.array(self.mesh.vertices)
        mask = np.zeros_like(x)
        self.save_uint8("mask", mask)

    def load_mask(self):
        mask = self.load_uint8("mask")
        return mask[..., 0] < 50

    def save_vertices_and_normals(self):
        vert = np.array(self.mesh.vertices)
        vert = (vert - self.box_min) / self.box_size
        norm = np.array(self.mesh.vertex_normals)
        norm = (norm + 1) / 2.0
        self.save_float("vertices", vert)
        self.save_float("normals", norm)

    def load_vertices_and_normals(self):
        vert = self.load_float("vertices")
        norm = self.load_float("normals")
        vert = vert * self.box_size + self.box_min
        norm = norm * 2.0 - 1
        return vert, norm

    def save_float(self, tag, arr):
        vs = split_float(arr)
        for i, vi in enumerate(vs):
            self.save_uint8(tag + f".v{i}", vi)
        # print(v1[v1 < 255].min(), "~", v1[v1 < 255].max())

    def load_float(self, tag):
        vs = []
        for i in range(4):
            vi = self.load_uint8(tag + f".v{i}")
            vi[vi < 255] = vi[vi < 255] * self.multiplier
            vi[vi >= 255] = 0
            vs.append(vi)
        return join_float(*vs)

    def save_uint8(self, tag, arr, offset=0.001):

        def mat(_):
            normalized_arr = (arr + offset) / 256.0
            vis = trimesh.visual.color.ColorVisuals(self.mesh, vertex_colors=normalized_arr)
            return vis

        render_mesh_texture(self.mesh, mat, self.get_cache_path(tag), self.resolution, self.uv)

    def load_uint8(self, tag):
        img = cv2.imread(self.get_cache_path(tag))
        return img[..., [2, 1, 0]]

    def erode(self, tag, kernel_size=2):
        erode_image(self.get_cache_path(tag), kernel_size)
        erode_image(self.get_cache_path(tag), kernel_size)

    def get_cache_path(self, tag):
        return os.path.join(self.cache_dir, f"{tag}x{self.resolution}.png")


def tst_ntr():
    ntr = NeuSTextureRenderer()
    # x = torch.rand(1000, 3).cuda()
    # out = ntr.int_surf(x, x)
    # print(out.shape)

    ntr.render_texture("./dev/lego_uv3.obj", "neus.png")
    ntr.show_model("./dev/lego_uv3.obj", "neus.png")
    # erode_image("neus.png")


def split_float_step(v, factor=2):
    Q = 256 // factor
    assert Q * factor == 256
    assert v.min() >= 0 and v.max() < 1
    vq = np.floor(v * Q)                        # 0~127 (int)
    vf = v * Q - vq                             # 0~1
    v1 = (vq * factor + factor // 2).astype(np.uint8)     # 0~255 (odd)
    return v1, vf


def join_float_step(v1, vf, factor=2):
    Q = 256 // factor
    assert Q * factor == 256
    assert v1.min() >= 0 and v1.max() < 256
    assert vf.min() >= 0 and vf.max() < 1
    vq = v1 // factor                           # 0~127 (int)
    return (vq + vf) / Q


def split_float(v):
    v1, v = split_float_step(v, 8)
    v2, v = split_float_step(v, 8)
    v3, v = split_float_step(v, 8)
    v4 = (v * 256).astype(np.uint8)            # 0~255
    return v1, v2, v3, v4


def join_float(v1, v2, v3, v4):
    v = v4 / 256.0
    v = join_float_step(v3, v, 8)
    v = join_float_step(v2, v, 8)
    v = join_float_step(v1, v, 8)
    return v


def tst_split_and_join():
    v = np.random.rand(10000, 3)
    vs = split_float(v)
    v_ = join_float(*vs)
    print(np.abs(v - v_).mean())


if __name__ == '__main__':
    # ntm = NeuSTextureModel("cache")
    # arr = ((np.array(ntm.mesh.vertices) + 1.5) / 3 * 255).astype(np.uint8)
    # ntm.save_uint8("tst", arr)
    # arr2 = ntm.load_uint8("tst")
    # ntm.save_uint8("tst", arr, offset=-0.00)
    # arr3 = ntm.load_uint8("tst")
    # print(arr2 - arr3)
    # print(np.abs(arr2 - arr3).sum())

    ntm = NeuSTextureModel("cache/lego_quad.ply", [-1.5, 1.5])
    # arr = (np.array(ntm.mesh.vertices) + 1.5) / 6
    # ntm.save_float("tst", arr)
    # arr2 = ntm.load_float("tst")
    # ntm.save_float("tst", arr)
    # arr3 = ntm.load_float("tst")
    # # print(arr2 - arr3)
    # print(np.abs(arr2 - arr3).sum())

    # ntm.erode("tst.v1")
    tst_split_and_join()

    # ones = np.ones_like(ntm.mesh.vertices)
    # ones[:1000, 0] = 0.11
    # ones[:1000, 1] = 0.21
    # ones[:1000, 2] = 0.31
    # ones[1000:-5000, 0] = 0.25
    # ones[1000:-5000, 1] = 0.55
    # ones[1000:-5000, 2] = 0.75
    # ones[-5000:, 0] = 0.89
    # ones[-5000:, 1] = 0.63
    # ones[-5000:, 2] = 0.42
    # ntm.save_float("tst", ones)
    # res = ntm.load_float("tst")
    # print(res)



