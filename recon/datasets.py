# coding=utf-8
# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different datasets implementation plus a general port for all the datasets."""
INTERNAL = False  # pylint: disable=g-statement-before-imports
import json
import os
from os import path
import queue
from re import S
from tarfile import POSIX_MAGIC
import threading
import warnings

if not INTERNAL:
    import cv2  # pylint: disable=g-import-not-at-top
import jax
import numpy as np
from PIL import Image
from nerf_sh.nerf import utils
from nerf_sh.nerf import depth_utils
from llff.poses import pose_utils
from tqdm import tqdm


def get_dataset(split, args):
    return dataset_dict[args.dataset](split, args)


def convert_to_ndc(origins, directions, focal, w, h, near=1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


class Dataset(threading.Thread):
    """Dataset Base Class."""

    def __init__(self, split, args):
        super(Dataset, self).__init__()
        self.queue = queue.Queue(2)  # Set prefetch buffer to 3 batches.
        self.daemon = True
        self.split = split
        self.image_batching = False
        if split == "train":
            self._train_init(args)
        elif split == "test":
            self._test_init(args)
        else:
            raise ValueError(
                'the split argument should be either "train" or "test", set'
                "to {} here.".format(split)
            )
        self.batch_size = args.batch_size // jax.host_count()
        self.render_path = args.render_path
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next training batch or test example.

        Returns:
          batch: dict, has "pixels" and "rays".
        """
        x = self.queue.get()
        if self.split == "train":
            return utils.shard(x)
        else:
            return utils.to_device(x)

    def peek(self):
        """Peek at the next training batch or test example without dequeuing it.

        Returns:
          batch: dict, has "pixels" and "rays".
        """
        x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
        if self.split == "train":
            return utils.shard(x)
        else:
            return utils.to_device(x)

    def run(self):
        if self.split == "train":
            next_func = self._next_train
        else:
            next_func = self._next_test
        while True:
            self.queue.put(next_func())

    @property
    def size(self):
        return self.n_examples

    def __len__(self):
        return self.size

    def _train_init(self, args):
        """Initialize training."""
        self._load_renderings(args)
        self._generate_rays()
        
        # default as no img batching cause camera refine and latent code
        self.images = self.images.reshape([-1, self.resolution, 3])
        self.naive_rays = utils.tuple_map(
            lambda x: x.reshape([-1]),self.naive_rays
        )
        if self.masked:
            self.masks = self.masks.reshape([-1, self.resolution, 1])

    def _test_init(self, args):
        self._load_renderings(args)
        self._generate_rays()
        self.it = 0

    def _next_train(self):
        image_index = np.random.randint(0, self.n_examples, ())
        ray_indices = np.random.randint(
                0, self.resolution, (self.batch_size,)
            )
        batch_pixels = self.images[image_index][ray_indices]
        batch_naive_rays = utils.tuple_map(
            lambda x: x[ray_indices], self.naive_rays
        )
        if self.masked:
            batch_masks = self.masks[image_index][ray_indices]
        
        # if want to use another kind of rays, just add meshgrid_mask field in return Dict
        # like split "meshgrid" with [mask>0]
        return{
                "view_id": self.view_idxs[image_index], 
                "pixels": batch_pixels, # (B,3)
                "camera": self.camtoworlds[self.view_idxs[image_index]],
                "focal": self.focal,
                "meshgrid": batch_naive_rays, # (B,) x 2
                "w": np.array(self.SRC_W),
                "h": np.array(self.SRC_H),
                "offsetXY": self.offsets[image_index]
            }

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        if self.render_path:
            return {
                "rays": utils.namedtuple_map(lambda r: r[idx], self.render_rays),
                "bd": np.array([self.start_near,self.end_far]),
            }
        else:
            return {
                "pixels": self.images[idx], #(H,W,3)
                "rays": utils.namedtuple_map(lambda r: r[idx], self.rays),
                "bd": np.array([self.start_near,self.end_far],dtype=np.float32), #(2,)
                "end_far": np.array(self.end_far,dtype=np.float32), #(1,)
            }

    # TODO(bydeng): Swap this function with a more flexible camera model.
    def _generate_rays(self):
        """Generating rays for all images."""
        print(' Generating rays')
        self.rays = utils.generate_rays(self.w, self.h, self.focal, self.camtoworlds)
        # the naive rays is the meshgrid of camera screen, (H,W) x 2, cause by jax cannot accept dynamic size array like "new mesh(w,h)"
        self.naive_rays = utils.generate_naive_rays(self.w, self.h)
    
    def _get_near_far(self):
        raise NotImplementedError
    
    def _get_center_radius_StartNEAR_EndFAR(self):
        raise NotImplementedError
    
    def get_center_world_boundary_info(self):
        raise NotImplementedError
    
    def get_views_num(self):
        raise NotImplementedError
    
    def get_train_views_num(self):
        raise NotImplementedError
    
    def getsc2Rsc1(self):
        raise NotImplementedError


class Blender(Dataset):
    """Blender Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        if args.render_path:
            raise ValueError("render_path cannot be used for the blender dataset.")
        with utils.open_file(
            path.join(args.data_dir, "transforms_{}.json".format(self.split)), "r"
        ) as fp:
            meta = json.load(fp)
        images = []
        cams = []
        print(' Load Blender', args.data_dir, 'split', self.split)
        for i in tqdm(range(len(meta["frames"]))):
            frame = meta["frames"][i]
            fname = os.path.join(args.data_dir, frame["file_path"] + ".png")
            with utils.open_file(fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                if args.factor == 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA
                    )
                elif args.factor > 0:
                    raise ValueError(
                        "Blender dataset only supports factor=0 or 2, {} "
                        "set.".format(args.factor)
                    )
            cams.append(frame["transform_matrix"])
            if args.white_bkgd:
                mask = image[..., -1:]
                image = image[..., :3] * mask + (1.0 - mask)
            else:
                image = image[..., :3]
            images.append(image)
        self.images = np.stack(images, axis=0)
        self.h, self.w = self.images.shape[1:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0).astype(np.float32)
        camera_angle_x = float(meta["camera_angle_x"])
        self.focal = 0.5 * self.w / np.tan(0.5 * camera_angle_x)

        self.n_examples = self.images.shape[0]
    
    def _get_near_far(self):
        return 2., 6.


class LLFF(Dataset):
    """LLFF Dataset."""
    
    def _load_renderings(self, args):
        """Load images from disk."""
        self.data_dir = path.expanduser(args.data_dir)
        self.masked = args.masked
        self.sparsity_ray_prop = 0.
        self.recenter = True
        print(' Load LLFF', args.data_dir, 'split', self.split)
        
        # Load images.
        imgdir_suffix = ""
        if args.factor > 0:
            imgdir_suffix = "_{}".format(args.factor)
            factor = args.factor
        else:
            factor = 1
        self.factor = factor
        imgdir = path.join(args.data_dir, "images" + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError("Image folder {} doesn't exist.".format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in sorted(utils.listdir(imgdir))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
                images.append(image)
        images = np.stack(images, axis=-1)# GS: will moveaxis below, so axis=-1 is ok
        
        # Load masks if needed
        if self.masked:
            maskdir = path.join(args.data_dir, "b_masks" + imgdir_suffix)
            if not utils.file_exists(maskdir):
                raise ValueError("b_masks folder {} doesn't exist.".format(maskdir))
            maskfiles = [
                path.join(maskdir,f)
                for f in sorted(utils.listdir(maskdir))
                if f.endswith("PNG") or f.endswith("png")
            ]
            print("Debug::_load_renderings::the sequence of masks order is: ",maskfiles)
            masks = []
            for maskfile in maskfiles:
                with utils.open_file(maskfile,"rb") as maskin:
                    mask = np.array(Image.open(maskin), dtype=np.float32) / 255.0
                    if(len(mask.shape)>2):
                        mask = mask[...,0]
                    masks.append(mask[...,None])
            masks = np.stack(masks,axis = -1)
            if masks.shape[-1] != images.shape[-1]:
                raise ValueError("enumerated missmatching between {} and {}".format(maskdir,imgdir))
            print("images.shape,masks.shape:{},{}".format(images.shape,masks.shape))
            
            

        # Load poses and bds.
        info = utils.get_center_info_from_colmap(self.data_dir)
        poses = info['poses'].transpose([1, 2, 0])
        bds1 = info['bds1'].transpose([1, 0])
        bds2 = info['bds2'].transpose([1, 0])
        if poses.shape[-1] != images.shape[-1] or bds1.shape[-1] != images.shape[-1] or bds2.shape[-1] != images.shape[-1]:
            raise RuntimeError(
                "Mismatch between imgs {} and poses {}".format(
                    images.shape[-1], poses.shape[-1]
                )
            )

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1]) # update hw
        poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor # update f

        # Correct rotation matrix ordering and move variable dim to axis 0.
        # whatever, after this step, pose is aligned to opengl-like right hand coordinate
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
        )
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        if self.masked:
            masks = np.moveaxis(masks, -1, 0)
        bds1 = np.moveaxis(bds1, -1, 0).astype(np.float32)
        bds2 = np.moveaxis(bds2, -1, 0).astype(np.float32)

        # Rescale according to a default bd factor.
        # old version as global scale
        #### mins,maxs = pose_utils.scene_boundary_min_max(args.data_dir)
        #### print("dataset::mins:{},maxs:{}".format(mins,maxs))
        #### max_range = (maxs-mins).max()
        #### scale = np.pi/np.abs(max_range)
        # new version with rescale(sc2 moved to __call__ in nerf) and recenter(done here)
        self.sc2Rsc1 = info['sc2']/info['sc1']
        scale = info['sc1']
        self.center = info['center'] - info['center']
        poses[:, :3, 3] -= info['center']
        poses[:, :3, 3] *= scale
        bds1 *= scale
        bds2 *= scale
        #### self.near = np.ndarray.min(bds) *0.9# GS
        #### self.far = np.ndarray.max(bds) * 1.# GS
        self.start_near = scale * info['uni_bd'][0]
        self.end_far = scale * info['uni_bd'][1]
        self.radius = scale * info['radius']

        # Recenter poses.
        # poses = self._recenter_poses(poses)

        # Generate a spiral/spherical ray path for rendering videos.
        if args.spherify and self.split == "test":
            self._generate_spherical_poses(poses, self.center)
            self.spherify = True
        else:
            self.spherify = False
        if not args.spherify and self.split == "test":
            raise NotImplementedError("no longer support spiral novel view now")
            # self._generate_spiral_poses(poses, bds) # no longer suit
        self.camtoworlds_allset = poses[:, :3, :4] # maintain all poses for some usage(extract) 
        
        # Select the split.
        i_test = np.arange(images.shape[0])[:: args.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if i not in i_test]
        )
        if len(args.train_views)>0:
            i_train = np.array(args.train_views)
        if len(args.test_views)>0:
            if args.test_views[0]>=0:
                i_test = np.array(args.test_views)
            else:
                i_test = []
        
        if self.split == "train":
            indices = i_train
            print("llff train views:{}".format(indices))
        else:
            indices = i_test
            print("llff test views:{}".format(indices))
        images = images[indices]
        poses = poses[indices]
        bds1 = bds1[indices]
        bds2 = bds2[indices]
        if self.masked:
            self.masks = masks[indices]
        
        self.images = images
        self.camtoworlds = poses[:, :3, :4]
        self.bds1 = bds1
        self.bds2 = bds2
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.SRC_H, self.SRC_W = self.h, self.w
        self.resolution = self.h * self.w
        self.views_num = indices.shape[-1]
        self.train_views_num = i_train.shape[-1]
        if args.render_path:
            self.n_examples = self.render_poses.shape[0]
        else:
            self.n_examples = images.shape[0]
        self.view_idxs = np.array(list(range(self.n_examples))) # 1-1 mapping to be Compatible with patch learning
        self.offsets = np.zeros((self.n_examples,2)) # 1-1 mapping to be Compatible with patch learning
        
        # trans z's center to 0
        # old version based on naive recenter method by nerf
        # mins,maxs = self.getWorldBoundary()
        # mid_z = (mins[2]+maxs[2])/2.0
        # poses[:, 2, 3] -= mid_z
        # self.camtoworlds = poses[:, :3, :4]
        # self.all_poses[:, 2, 3] -= mid_z
        # if self.split == "test":
        #     self.render_poses[:, 2, 3] -= mid_z

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        if self.split == "test":
            n_render_poses = self.render_poses.shape[0]
            self.camtoworlds = np.concatenate(
                [self.render_poses, self.camtoworlds], axis=0
            )

        super()._generate_rays()
        
        # GS: shouldn't be ndc
        # if not self.spherify:
        #     ndc_origins, ndc_directions = convert_to_ndc(
        #         self.rays.origins, self.rays.directions, self.focal, self.w, self.h
        #     )
        #     self.rays = utils.Rays(
        #         origins=ndc_origins,
        #         directions=ndc_directions,
        #         viewdirs=self.rays.viewdirs,
        #     )

        # Split poses from the dataset and generated poses
        if self.split == "test":
            self.camtoworlds = self.camtoworlds[n_render_poses:]
            split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            split0, split1 = zip(*split)
            self.render_rays = utils.Rays(*split0)
            self.rays = utils.Rays(*split1)
            
    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable "focus depth" for this dataset.
        close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
        dt = 0.75
        mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.0])
        hwf = c2w_path[:, 4:5]
        zrate = 0.5
        for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(
                c2w[:3, :4],
                (
                    np.array(
                        [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
                    )
                    * rads
                ),
            )
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]
    
    def _generate_spherical_poses(self,poses,center):
        """Generate a 360 degree spherical path for rendering."""

        up = self._normalize(-(poses[:, :3, 3] - center).mean(0))
        # up = _normalize(np.percentile((poses[:, :3, 3] - center),50.0,axis=0))
        right = self._normalize(np.cross([0.1,0.2,0.3],up))
        back = self._normalize(np.cross(right,up))
        zh = np.dot((poses[:, :3, 3] - center).mean(0),up)

        orbit_sphere_rad = np.linalg.norm(poses[:, :3, 3] - center,axis=-1).max()
        radcircle = np.sqrt(orbit_sphere_rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0.0, 2.0 * np.pi, 120):
            # camorigin = np.array([radcircle * np.cos(th) * right, radcircle * np.sin(th) * back, zh * up])+center
            camorigin = radcircle * np.cos(th) * right + radcircle * np.sin(th) * back + zh * up + center
            vec2 = self._normalize(camorigin-center) 
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate(
            [
                new_poses,
                np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape),
            ],
            -1,
        )
        if self.split == "test":
            self.render_poses = new_poses[:, :3, :4]
        return up

    
    def _get_near_far(self):
        return self.near, self.far
    
    def _get_center_radius_StartNEAR_EndFAR(self):
        return np.array(self.center,dtype=np.float32), np.array(self.radius,dtype=np.float32), np.array(self.start_near,dtype=np.float32), np.array(self.end_far,dtype=np.float32)
    
    def get_center_world_boundary_info(self):
        if self.recenter:
            return {
                "mins": np.zeros((3,),dtype=np.float32) - self.radius, 
                "maxs": np.zeros((3,),dtype=np.float32) + self.radius,
                "center": np.zeros((3,),dtype=np.float32),
                "radius": np.array([self.radius],dtype=np.float32)
            }
        else:
            raise NotImplementedError("the dataset must set origin as 0 now!")
    
    def get_views_num(self):
        return self.views_num
    def get_train_views_num(self):
        return self.train_views_num

class NSVF(Dataset):
    """NSVF Generic Dataset."""

    def _load_renderings(self, args):
        """Load images from disk."""
        if args.render_path:
            raise ValueError("render_path cannot be used for the NSVF dataset.")
        args.data_dir = path.expanduser(args.data_dir)
        K : np.ndarray = np.loadtxt(path.join(args.data_dir, "intrinsics.txt"))
        pose_files = sorted(os.listdir(path.join(args.data_dir, 'pose')))
        img_files = sorted(os.listdir(path.join(args.data_dir, 'rgb')))

        if self.split == 'train':
            pose_files = [x for x in pose_files if x.startswith('0_')]
            img_files = [x for x in img_files if x.startswith('0_')]
        elif self.split == 'val':
            pose_files = [x for x in pose_files if x.startswith('1_')]
            img_files = [x for x in img_files if x.startswith('1_')]
        elif self.split == 'test':
            test_pose_files = [x for x in pose_files if x.startswith('2_')]
            test_img_files = [x for x in img_files if x.startswith('2_')]
            if len(test_pose_files) == 0:
                test_pose_files = [x for x in pose_files if x.startswith('1_')]
                test_img_files = [x for x in img_files if x.startswith('1_')]
            pose_files = test_pose_files
            img_files = test_img_files

        images = []
        cams = []

        cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))

        assert len(img_files) == len(pose_files)
        print(' Load NSVF', args.data_dir, 'split', self.split, 'num_images', len(img_files))
        for img_fname, pose_fname in tqdm(zip(img_files, pose_files), total=len(img_files)):
            img_fname = path.join(args.data_dir, 'rgb', img_fname)
            with utils.open_file(img_fname, "rb") as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
            cam_mtx = np.loadtxt(path.join(args.data_dir, 'pose', pose_fname)) @ cam_trans
            cams.append(cam_mtx)  # C2W
            if image.shape[-1] == 4:
                # Alpha channel available
                if args.white_bkgd:
                    mask = image[..., -1:]
                    image = image[..., :3] * mask + (1.0 - mask)
                else:
                    image = image[..., :3]
            if args.factor > 1:
                [rsz_h, rsz_w] = [hw // args.factor for hw in image.shape[:2]]
                image = cv2.resize(
                    image, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA
                )

            images.append(image)
        self.images = np.stack(images, axis=0)
        self.n_examples, self.h, self.w = self.images.shape[:3]
        self.resolution = self.h * self.w
        self.camtoworlds = np.stack(cams, axis=0).astype(np.float32)
        # We assume fx and fy are same
        self.focal = (K[0, 0] + K[1, 1]) * 0.5
        if args.factor > 1:
            self.focal /= args.factor

class LLFFPatch(Dataset):
    """LLFF Dataset."""
    
    def _load_renderings(self, args):
        """Load images from disk."""
        self.data_dir = path.expanduser(args.data_dir)
        self.recenter = True
        self.masked = False
        print(' Load LLFF PATCH to FINE TUNE', args.data_dir)
        
        # Load scene info, patch cliped from original resolution.
        with open(os.path.join(args.data_dir,'patches.json')) as fp:
            meta = json.load(fp)
        self.SRC_W, self.SRC_H = meta["SRC_WH"]
        info = utils.get_center_info_from_colmap(self.data_dir)
        
        ## 1 Load poses.
        poses = info['poses'].transpose([1, 2, 0])
        poses[:2, 4, :] = np.array((self.SRC_H,self.SRC_W)).reshape([2, 1]) # update hw
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
        )
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        
        ## 2 rescale
        scale = info['sc1']
        self.center = info['center'] - info['center']
        poses[:, :3, 3] -= info['center']
        poses[:, :3, 3] *= scale
        self.start_near = scale * info['uni_bd'][0]
        self.end_far = scale * info['uni_bd'][1]
        self.radius = scale * info['radius']
        
        # Select the view num in super dataset.
        i_test = np.arange(poses.shape[0])[:: args.llffhold]
        i_train = np.array(
            [i for i in np.arange(int(poses.shape[0])) if i not in i_test]
        )
        
        # Load patches, patch cliped from original resolution.
        factor = 1
        self.factor = factor
        name_to_viewidx = {}
        idx = 0
        for name in info["cam_order"]:
            name_to_viewidx[os.path.splitext(name)[0]] = idx
            idx += 1
        images = []
        offsets = []
        view_idxs = []
        for i in range(len(meta["patches"])):
            patch = meta["patches"][i]
            p_path = os.path.join(args.data_dir,"refine_patches",patch['src_img_name'],patch['patch_name']+'.jpg')
            p_img = np.array(Image.open(p_path), dtype=np.float32) / 255.0
            p_offset = np.array(patch['offset_XY'],dtype=np.float32)
            p_vidx = name_to_viewidx[patch['src_img_name']]
            images.append(p_img)
            offsets.append(p_offset)
            view_idxs.append(p_vidx)
        images = np.stack(images,axis=0)
        offsets = np.stack(offsets,axis=0)
        view_idxs = np.array(view_idxs)
        
        # batch info for training
        self.images = images
        self.offsets = offsets
        self.view_idxs = view_idxs
        self.camtoworlds = poses[:, :3, :4]
        self.train_views_num = i_train.shape[-1]
        self.views_num = i_train.shape[-1]
        self.focal = poses[0, -1, -1]
        self.h, self.w = images.shape[1:3]
        self.resolution = self.h * self.w
        self.n_examples = images.shape[0]

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        print(' LLFF PATCH Generating rays')
        self.rays = utils.generate_patch_rays(self.w, self.h, self.SRC_W, self.SRC_H, self.focal, self.camtoworlds, self.offsets, self.view_idxs)
        self.naive_rays = utils.generate_naive_rays(self.w, self.h)

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    
    def _get_near_far(self):
        raise NotImplementedError
    
    def _get_center_radius_StartNEAR_EndFAR(self):
        return np.array(self.center,dtype=np.float32), np.array(self.radius,dtype=np.float32), np.array(self.start_near,dtype=np.float32), np.array(self.end_far,dtype=np.float32)
    
    def get_center_world_boundary_info(self):
        if self.recenter:
            return {
                "mins": np.zeros((3,),dtype=np.float32) - self.radius, 
                "maxs": np.zeros((3,),dtype=np.float32) + self.radius,
                "center": np.zeros((3,),dtype=np.float32),
                "radius": np.array([self.radius],dtype=np.float32)
            }
        else:
            raise NotImplementedError("the dataset must set origin as 0 now!")
    
    def get_views_num(self):
        return self.views_num
    def get_train_views_num(self):
        return self.train_views_num


dataset_dict = {
    "blender": Blender,
    "llff": LLFF,
    "nsvf": NSVF,
    "llffpatch": LLFFPatch,
}
