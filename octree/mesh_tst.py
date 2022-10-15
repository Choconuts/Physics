import torch
import numpy as np
import trimesh
from neus_tst import *

nerf_octree = NeuSOctree()

mesh = trimesh.load(r"G:\WorkSpace\nerf\logs\lego-neus-return\meshes\mesh_050000.ply")

vert = torch.tensor(np.array(mesh.vertices), device="cuda")

should_divide = torch.zeros_like(nerf_octree.octree.non_leaf)[..., 0]
idx = nerf_octree.octree.query(vert)[..., 0]
should_divide = torch.index_put(should_divide, (idx,), torch.ones_like(idx), False)

box = nerf_octree.octree.boxes[0]
octree0 = Octree(box[:3], box[3:])


def sub_divide(boxes):
    size = boxes[..., 3:]
    center = boxes[..., :3] + size * 0.5
    idx = nerf_octree.octree.query(center)[..., 0]
    prior_boxes = nerf_octree.octree.boxes[idx]
    not_divide = (prior_boxes[..., 3:] >= size).prod(-1).bool()
    return torch.logical_or(~not_divide, should_divide[idx])


octree0.build(sub_divide, 10, 2)


octree = Octree(box[:3], box[3:])


should_divide2 = torch.zeros_like(nerf_octree.octree.non_leaf)[..., 0]
idx = octree0.query(vert)[..., 0]
should_divide2 = torch.index_put(should_divide2, (idx,), torch.ones_like(idx), False)


def sub_divide2(boxes):
    size = boxes[..., 3:]
    center = boxes[..., :3] + size * 0.5
    idx = octree0.query(center)[..., 0]
    prior_boxes = octree0.boxes[idx]
    not_divide = (prior_boxes[..., 3:] > size).prod(-1).bool()
    return torch.logical_and(~not_divide, should_divide2[idx])[..., None]


octree.build(sub_divide2, 15, 2)


class A:

    @ui(opt)
    def show_boxes(self):
        select_leaf = octree0.boxes[..., 3] <= octree0.boxes[..., 3].min() + 1e-4
        boxes = octree0.boxes[select_leaf]
        box_min, box_size = torch.split(boxes, [3, 3], -1)
        x_mask = torch.tensor([1, 0, 0.], device=box_min.device)
        y_mask = torch.tensor([0, 1, 0.], device=box_min.device)
        z_mask = torch.tensor([0, 0, 1.], device=box_min.device)
        xy_mask = torch.tensor([1, 1, 0.], device=box_min.device)
        yz_mask = torch.tensor([0, 1, 1.], device=box_min.device)
        xz_mask = torch.tensor([1, 0, 1.], device=box_min.device)
        pnts = [
            box_min,
            box_min + box_size * x_mask,
            box_min + box_size * y_mask,
            box_min + box_size * z_mask,
            box_min + box_size * xy_mask,
            box_min + box_size * yz_mask,
            box_min + box_size * xz_mask,
            box_min + box_size,
        ]
        pnts = torch.cat(pnts, 0)


        rays_o = torch.tensor([
            [0.9, 0.9, 0.9],
            [0.7, 0.6, 0.5]
        ], device="cuda")

        rays_d = torch.tensor([
            [-1, -1, -1.]
        ], device="cuda")

        t = no.cast(rays_o, rays_d)

        while True:
            if opt.changed:
                if opt.show:
                    visualize_field(pnts, scalars=pnts)
                visualize_field(rays_o, vectors=rays_d * t, len_limit=0)
            yield


def render(octree):
    no.octree = octree
    st = time.time()
    for i in range(20):
        rays_o, rays_d, rgb, mask = scene.sample_image(i, -1)
        # print(rays_o.shape, rays_d.shape, rgb.shape, mask.shape)
        t = no.cast(rays_o, rays_d)
        p = rays_o + t * rays_d
        with torch.no_grad():
            a = no.nerf(p, rays_d)
            rgb = no.nerf.color(p, rays_d, rays_d, a[..., 1:])

        scene.save_image(rgb.view(*scene.data.img_res, 3), i)
        scene.save_gray_image(t.view(*scene.data.img_res) / 4, i)
    print("[Cost]", f"trace {20} images, cost {time.time() - st:.4f} seconds.")


print(should_divide.nonzero().numel())
print(idx.shape)

if __name__ == '__main__':
    scene = ObserveScene(20, dh=False)

    A().show_boxes()
    # render(octree0)
