import time
import numpy as np
import trimesh
import xatlas
from trimesh.visual import material
from PIL import Image
import cv2


def uvmap_tst():
    mesh = trimesh.load("lego_quad.ply")
    # vertices = np.array(mesh.vertices)
    # indices = np.array(mesh.faces)
    # atlas.add_mesh(vertices, indices)
    # atlas.generate(xatlas.ChartOptions())
    # vert, face, uv = atlas.get_mesh(0)
    # print(uv)
    # print(uv.shape)

    st = time.time()
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    xatlas.export("lego_uv4.obj", mesh.vertices[vmapping], indices, uvs)
    print("[Parameterize]", time.time() - st, "seconds")


def texture_tst():
    img = Image.open("tex.png")
    mesh = trimesh.load("lego_uv3.obj")
    tex_visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, image=img)
    # assert isinstance(mesh, trimesh.Trimesh)
    # mat = material.from_color(mesh.vertices)
    mesh.visual = tex_visual
    mesh.show()


def tex_paint():
    mesh = trimesh.load("lego_uv3.obj")
    uv = np.array(mesh.visual.uv)
    uvw = np.concatenate([uv[..., 1:], uv[..., :1
                                       ], np.zeros_like(uv[..., :1])], -1)
    vt = np.array(mesh.vertices)
    color = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vt / 3 + 0.5)
    mesh.vertices = uvw
    mesh.visual = color
    tex = color.to_texture()
    print(tex)
    scene = trimesh.scene.scene.Scene(mesh)
    img = scene.save_image((1024, 1024))
    with open("save.png", "bw") as fp:
        fp.write(img)

    # img = Image.open("tex.png")
    # tex_visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)
    # mesh.visual = tex_visual
    #
    # scene = trimesh.scene.scene.Scene(mesh)
    # img = scene.save_image((1417, 1417))
    # with open("save3.png", "bw") as fp:
    #     fp.write(img)
    # mesh.show()


def reload():
    mesh = trimesh.load("lego_uv3.obj")
    uv = np.array(mesh.visual.uv)
    uvw = np.concatenate([uv, np.zeros_like(uv[..., :1])], -1)
    vt = np.array(mesh.vertices)
    color = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vt / 3 + 0.5)
    mesh.vertices = uvw
    mesh.show()
    mesh.visual = color
    tex = color.to_texture()
    print(tex)
    img = Image.open("save3.png")
    tex_visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=img)
    mesh.visual = tex_visual
    mesh.show()
    # mesh.vertices = vt
    # mesh.show()


def expand():
    img = cv2.imread("save.png")
    kernel = np.ones((32, 32), np.uint8)
    img = cv2.erode(img, kernel)
    cv2.imwrite("save2.png", img)


def normal_tst():
    img = Image.open("tex.png")
    mesh = trimesh.load("lego_uv3.obj")

    vt = np.array(mesh.vertices)
    nm = np.array(mesh.vertex_normals)
    color = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=nm * 0.5 + 0.5)       # vt / 3 + 0.5)

    # origin_normal = mesh.vertex_normals
    # vs = np.array(mesh.vertices)
    # fs = np.array(mesh.faces)
    # e1 = vs[fs[..., 1]] - vs[fs[..., 0]]
    # e2 = vs[fs[..., 2]] - vs[fs[..., 1]]
    # n = np.cross(e1, e2)
    # n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    # print(n.shape)
    #
    # # nm = np.array(mesh.face_normals)
    # color = trimesh.visual.color.ColorVisuals(mesh, face_colors=n * 0.5 + 0.5)       # vt / 3 + 0.5)
    mesh.visual = color
    mesh.show()


if __name__ == '__main__':
    normal_tst()



