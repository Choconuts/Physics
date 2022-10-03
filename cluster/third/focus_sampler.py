import torch
import torch.nn.functional as F

from utils import rend_util


def inv_camera_params(locations, pose_inv, cam_loc, intrinsics):
    batch_size, num_samples, _ = locations.shape
    ray_dirs = F.normalize(locations - cam_loc[:, None, :], dim=-1)

    norm_inv = pose_inv @ torch.cat([ray_dirs + cam_loc[:, None, :], torch.ones_like(ray_dirs[..., -1:])], -1).permute(0, 2, 1)

    z_neg = -norm_inv[:, 2:3, :]
    ppc_inv = norm_inv / torch.where(z_neg != 0, z_neg, 1e-5)
    ppc_inv[:, 1:3, :] *= -1
    uv_inv = intrinsics @ ppc_inv[:, :3, :]
    uv = uv_inv.permute(0, 2, 1)[..., :2]

    return uv, ray_dirs


class FocusSampler:

    def __init__(self, dataset):
        self.dataset = dataset
        pose = torch.stack(self.dataset.pose_all, 0)
        if pose.shape[1] == 7:  # In case of quaternion vector representation
            cam_loc = pose[:, 4:]
            R = rend_util.quat_to_rot(pose[:, :4])
            p = torch.eye(4).repeat(pose.shape[0], 1, 1).cuda().float()
            p[:, :3, :3] = R
            p[:, :3, 3] = cam_loc
        else:  # In case of pose matrix representation
            cam_loc = pose[:, :3, 3]
            p = torch.eye(4).repeat(pose.shape[0], 1, 1).cuda().float()
            p[:, :3, :4] = pose[:, :3, :4]
        self.pose_inv = p.inverse()
        self.cam_loc = cam_loc.cuda()

        shape = [self.dataset.n_cameras, self.dataset.img_res[0], self.dataset.img_res[1], -1]
        self.images = torch.stack(self.dataset.rgb_images, 0).view(shape).permute(0, 3, 1, 2).cuda().float()
        self.masks = torch.cat(self.dataset.object_masks, 0).view(shape).permute(0, 3, 1, 2).cuda().float()
        self.intrinsics = torch.stack(self.dataset.intrinsics_all, 0).cuda()

        self.img_size = torch.tensor([self.dataset.img_res[1], self.dataset.img_res[0]]).cuda()

    def sample_images(self, uv):
        uv = (uv[:, None] / self.img_size) * 2 - 1
        color = F.grid_sample(self.images, uv)
        return color.permute(0, 2, 3, 1)[:, 0]

    def sample_masks(self, uv):
        uv = (uv[:, None] / self.img_size) * 2 - 1
        color = F.grid_sample(self.masks, uv)
        return color.permute(0, 2, 3, 1)[:, 0] > 0.5

    def scatter_sample(self, x):
        assert len(x.shape) == 2
        x = x[None].expand(115, -1, -1)

        uv, ray_dirs = inv_camera_params(x, self.pose_inv, self.cam_loc, self.intrinsics)
        rgb = self.sample_images(uv)

        # TODO: add valid mask
        uv_valid = torch.logical_and(uv >= 0, uv < torch.tensor(self.dataset.img_res).to(uv.device)).prod(-1).bool()
        if uv_valid.any() and self.masks.numel() > 0:
            uv_valid[uv_valid.clone()] = (self.sample_masks(uv[uv_valid][None])).squeeze()

        sample = {
            # "pose": torch.stack(poses, 0).cuda(),
            # "intrinsics": torch.stack(intrinsics_list, 0).cuda(),
            "object_mask": uv_valid,
            "uv": uv,
            "view_dir": ray_dirs,
        }

        ground_truth = {
            "rgb": rgb,
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
    from cluster.third.syn_dataset import SynDataset

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
