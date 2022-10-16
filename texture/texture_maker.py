import time
import numpy as np
import trimesh
import xatlas
from trimesh.visual import material
from PIL import Image
import cv2


def gen_uv_map(mesh_path, out_path=None):
    mesh = trimesh.load(mesh_path)
    st = time.time()
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    if out_path is None:
        out_path = ".".join(mesh_path.split(".")[:-1]) + ".obj"
    xatlas.export(out_path, mesh.vertices[vmapping], indices, uvs)
    print("[Parameterize]", time.time() - st, "seconds")


def render_texture(mesh_path, material_fn, out_path, resolution=1024, erode=4):
    mesh = trimesh.load(mesh_path)
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
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        combine = cv2.erode(combine, kernel)
    cv2.imwrite(out_path, combine)


if __name__ == '__main__':

    def mat(mesh):
        vt = np.array(mesh.vertices)
        return trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vt / 3 + 0.5)

    def mat(mesh):
        vt = np.array(mesh.vertex_normals)
        return trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vt / 2 + 0.5)

    render_texture("dev/lego_uv3.obj", mat, "tex.png")
    img = Image.open("tex.png")
    mesh = trimesh.load("dev/lego_uv3.obj")
    tex_visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, image=img)
    mesh.visual = tex_visual
    mesh.show()