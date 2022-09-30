import torch
import torch.nn.functional as F
import numpy as np

from utils import rend_util


def inv_camera_params(locations, pose, intrinsics):
    if pose.shape[1] == 7: # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = rend_util.quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :4] = pose[:, :3, :4]

    batch_size, num_samples, _ = locations.shape
    ray_dirs = F.normalize(locations - cam_loc, dim=2)

    norm_inv = p.inverse() @ torch.cat([ray_dirs + cam_loc, torch.ones_like(ray_dirs[..., -1:])], -1).permute(0, 2, 1)

    z_neg = -norm_inv[:, 2:3, :]
    ppc_inv = norm_inv / torch.where(z_neg != 0, z_neg, 1e-5)
    ppc_inv[:, 1:3, :] *= -1
    uv_inv = intrinsics @ ppc_inv[:, :3, :]
    uv = uv_inv.permute(0, 2, 1)[..., :2]

    return uv, ray_dirs


class FocusSampler:

    def __init__(self, dataset):
        self.dataset = dataset

    def scatter_sample(self, x):
        assert len(x.shape) == 2
        poses = []
        intrinsics_list = []
        object_masks = []
        rgbs = []
        uvs = []
        view_dirs = []

        def sample_uv(tex, uv):
            uv = (uv[None] / torch.tensor(self.dataset.img_res).to(uv.device)) * 2 - 1
            color = F.grid_sample(
                tex.view(1, self.dataset.img_res[0], self.dataset.img_res[1], -1).permute(0, 3, 1, 2).cuda().float(), uv)
            return color.permute(0, 2, 3, 1)[0]

        for i in range(self.dataset.n_cameras):
            intrinsics = self.dataset.intrinsics_all[i]
            pose = self.dataset.pose_all[i]
            uv, ray_dirs = inv_camera_params(x[None], pose[None].to(x.device), intrinsics[None].to(x.device))
            object_mask = self.dataset.object_masks[i]
            rgb = self.dataset.rgb_images[i]
            poses.append(pose)
            intrinsics_list.append(intrinsics)

            # TODO: add valid mask
            uv_valid = torch.logical_and(uv >= 0, uv < torch.tensor(self.dataset.img_res).to(uv.device)).prod(-1).bool()
            uv_valid[uv_valid.clone()] = (sample_uv(object_mask, uv[uv_valid][None]) > 0.5).squeeze()
            object_masks.append(uv_valid)

            uvs.append(uv)
            rgbs.append(sample_uv(rgb, uv))
            view_dirs.append(ray_dirs)

        sample = {
            "pose": torch.stack(poses, 0).cuda(),
            "intrinsics": torch.stack(intrinsics_list, 0).cuda(),
            "object_mask": torch.cat(object_masks, 0),
            "uv": torch.cat(uvs, 0),
            "view_dir": torch.cat(view_dirs, 0),
        }

        ground_truth = {
            "rgb": torch.cat(rgbs, 0),
        }

        """
        M: dataset image num (or maximum num), N: sample num
        x: [N, 3] (all cuda only)
        {
            pose: [M, 4, 4]
            intrinsics: [M, 3, 3]
            object_mask: [M, N, 1] (bool)
            uv: [M, 4, 4]
            view_dirs: [M, N, 3]
        }
        {
            rgb: [M, N, 3]
        }
        """

        if sample["uv"][sample["object_mask"]].isnan().any():
            raise RuntimeError("uv valid is nan")

        return sample, ground_truth


if __name__ == '__main__':
    from syn_dataset import SynDataset

    data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 10)
    focus_sampler = FocusSampler(data)

    sample, gt = focus_sampler.scatter_sample(torch.rand(512, 3).cuda())
    for s in sample:
        print(s, sample[s].shape, sample[s].dtype, sample[s].device)
    print("rgb", gt["rgb"].shape, gt["rgb"].dtype, gt["rgb"].device)

    d, o = rend_util.get_camera_params(sample["uv"], sample["pose"], sample["intrinsics"])
    print("rays", d.shape, o.shape)

    diff = (d - sample["view_dir"]) ** 2
    print("diff", diff.mean().item())
