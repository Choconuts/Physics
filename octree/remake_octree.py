from mesh_to_sdf import sample_sdf_near_surface
from octree import Octree
from cluster.observe import *
import trimesh
import pyrender
import numpy as np


def fine_sdf(x):
    return (x - 0.5).norm(dim=-1) - 0.3


def octree_from_sdf(sdf, bound=((0, 0, 0), (1, 1, 1))):
    box = bound[0], [bound[1][i] - bound[0][i] for i in range(3)]
    coarse = Octree(*box)

    def coarse_fn(boxes):
        size = boxes[..., 3:]
        center = boxes[..., :3] + size * 0.5
        return sdf(center).abs() < size.norm(dim=-1) * 0.5

    coarse.build(coarse_fn, 10, 5)
    centers = coarse.boxes[..., :3] + coarse.boxes[..., 3:] * 0.5
    sdf_values = sdf(centers)

    return coarse


def show_octree(octree):
    class A:

        @ui(opt)
        def show_boxes(self):
            select_leaf = octree.boxes[..., 3] <= octree.boxes[..., 3].min() + 1e-4
            boxes = octree.boxes[select_leaf]
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
            visualize_field(pnts, scalars=pnts)

    A().show_boxes()


def show_mesh(mesh):
    class A:
        @ui(opt)
        def show_boxes(self):
            vert = np.array(mesh.vertices)
            visualize_field(vert, vert)
    A().show_boxes()


if __name__ == '__main__':
    show_octree(octree_from_sdf(fine_sdf))

