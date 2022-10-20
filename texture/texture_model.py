import os
import time
import torch
from torch import nn
import numpy as np
import xatlas
import imageio
import trimesh
from trimesh.visual import material
from rasterizor import texture_rasterizor


def gen_uv_map(mesh_path, out_path=None):
    mesh = trimesh.load(mesh_path)
    st = time.time()
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    if out_path is None:
        out_path = ".".join(mesh_path.split(".")[:-1]) + ".obj"
    xatlas.export(out_path, mesh.vertices[vmapping], indices, uvs)
    print("[Parameterize]", time.time() - st, "seconds")


class TextureModel(nn.Module):

    def __init__(self, mesh_path, resolution=1024):
        super(TextureModel, self).__init__()
        self.cache_dir = self.init_cache_dir(mesh_path)
        self.mesh, (self.box_min, self.box_size), self.uv = self.init_mesh(mesh_path)
        self.resolution = resolution
        self.vert, self.norm, self.mask = self.init_basics()

    def init_mesh(self, mesh_path):
        mesh = trimesh.load(mesh_path)
        if not hasattr(mesh.visual, "uv"):
            new_name = ".".join(os.path.basename(mesh_path).split(".")[:-1]) + ".obj"
            new_path = os.path.join(self.cache_dir,new_name)
            if not os.path.exists(new_path):
                print("[Parameterize] generate uv map")
                gen_uv_map(mesh_path, new_path)
            mesh = trimesh.load(new_path)

        vert = np.array(mesh.vertices)
        box_min = vert.min(axis=0, initial=0.0) - 1e-2
        box_max = vert.max(axis=0, initial=0.0) + 1e-2

        return mesh, (box_min, box_max - box_min), np.array(mesh.visual.uv)

    def init_cache_dir(self, mesh_path):
        cache_dir = ".".join(os.path.basename(mesh_path).split(".")[:-1]) + ".cache"
        cache_dir = os.path.join(os.path.dirname(mesh_path), cache_dir)
        if os.path.exists(cache_dir):
            print("[Use Cache]", cache_dir)
        else:
            os.makedirs(cache_dir)
            print("[Make Cache]", cache_dir)

        return cache_dir

    def init_basics(self):
        if not os.path.exists(self.get_cache_path("vert")):
            print("[Cache] generate vertices, normals and masks")
            vert = np.array(self.mesh.vertices)
            norm = np.array(self.mesh.vertex_normals)
            mask = np.ones_like(vert)
            self.save_float("vert", vert)
            self.save_float("norm", norm)
            self.save_float("mask", mask)

        vert = self.load_float("vert")
        norm = self.load_float("norm")
        mask = self.load_float("mask")
        return vert[..., :3], norm[..., :3], mask[..., 0] > 0.5

    def save_float(self, tag, arr, resolution=-1):
        if resolution < 0:
            resolution = self.resolution
        with texture_rasterizor(resolution) as tex_render:
            data = tex_render(self.uv, self.mesh.faces, arr)
            imageio.imwrite(self.get_cache_path(tag), data)

    def load_float(self, tag):
        return imageio.imread(self.get_cache_path(tag))

    def get_cache_path(self, tag, ext="exr", resolution=-1):
        if resolution < 0:
            resolution = self.resolution
        return os.path.join(self.cache_dir, f"{tag}x{resolution}.{ext}")


if __name__ == '__main__':
    from interface import visualize_field, ui, Inputable
    ntm = TextureModel("cache/lego_quad.ply")

    class A:

        @ui
        def show_points(self):
            visualize_field(ntm.vert.reshape([-1, 3]), scalars=ntm.norm.reshape([-1, 3]))
            while True:
                yield

    A().show_points()

