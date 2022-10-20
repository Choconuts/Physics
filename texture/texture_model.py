import os
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import trimesh
import xatlas
import imageio

from rasterizor import texture_rasterizor


def gen_uv_map(mesh_path, out_path=None):
    mesh = trimesh.load(mesh_path)
    st = time.time()
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    if out_path is None:
        out_path = ".".join(mesh_path.split(".")[:-1]) + ".obj"
    xatlas.export(out_path, mesh.vertices[vmapping], indices, uvs)
    print("[Parameterize]", time.time() - st, "seconds")


class TextureCache:

    def __init__(self, mesh_path):
        super(TextureCache, self).__init__()
        self.cache_dir = self.init_cache_dir(mesh_path)
        self.mesh, (self.box_min, self.box_size), self.uv = self.init_mesh(mesh_path)

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

    def render_basics(self, resolution):
        """ Do not call this in OpenGL context !!! """
        if not os.path.exists(self.get_cache_path("vert", resolution)):
            print("[Cache] generate vertices, normals and masks")
            vert = np.array(self.mesh.vertices)
            norm = np.array(self.mesh.vertex_normals)
            mask = np.ones_like(vert)
            self.save_float("vert", resolution, vert)
            self.save_float("norm", resolution, norm)
            self.save_float("mask", resolution, mask)

    def load_basics(self, resolution):
        vert = self.load_float("vert", resolution)
        norm = self.load_float("norm", resolution)
        mask = self.load_float("mask", resolution)
        return vert[..., :3], norm[..., :3], mask[..., 0] > 0.5

    def save_float(self, tag, resolution, arr):
        with texture_rasterizor(resolution) as tex_render:
            data = tex_render(self.uv, self.mesh.faces, arr)
            imageio.imwrite(self.get_cache_path(tag, resolution), data)

    def load_float(self, tag, resolution):
        return imageio.imread(self.get_cache_path(tag, resolution))

    def get_cache_path(self, tag, resolution, ext="exr"):
        return os.path.join(self.cache_dir, f"{tag}x{resolution}.{ext}")


class PBRTextureModel(nn.Module):

    def __init__(self, res=1024):
        super(PBRTextureModel, self).__init__()
        self.albedo_metalic_roughness = nn.Parameter(torch.ones(1, 5, res, res) * 0.5)

    def forward(self, uv):
        init_shape = list(uv.shape[:-1]) + [-1]
        uv = uv.reshape(1, 1, -1, 2) * 2 - 1
        all_data = F.grid_sample(self.albedo_metalic_roughness, uv).reshape(5, -1).permute(1, 0)
        all_data = all_data.reshape(*init_shape)
        return torch.split(all_data, [3, 1, 1], -1)


def get_vert_norm_mask_maps(mesh_path, resolution=1024):
    tex_cache = TextureCache(mesh_path)
    tex_cache.render_basics(resolution)
    vert, norm, mask = tex_cache.load_basics(resolution)
    return torch.tensor(vert).float().cuda(), torch.tensor(norm).float().cuda(), torch.tensor(mask).float().cuda()


if __name__ == '__main__':
    from interface import visualize_field, ui
    tex_cache = TextureCache("cache/lego_quad.ply")
    tex_cache.render_basics(512)

    pbr = PBRTextureModel(512)
    albedo, metalic, roughness = pbr(torch.rand(512 * 512, 2))
    print(albedo.shape, metalic.shape, roughness.shape)

    vert, norm, mask = get_vert_norm_mask_maps("cache/hotdog_mc.ply")
    print(vert.shape, norm.shape, mask.shape)

    class A:

        @ui
        def show_points(self):
            vert, norm, mask = tex_cache.load_basics(512)
            visualize_field(vert.reshape([-1, 3]), scalars=norm.reshape([-1, 3]))
            while True:
                yield

    A().show_points()

