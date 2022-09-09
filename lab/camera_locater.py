
import torch
from torch import nn
from torch.autograd.functional import jacobian
from functional import Simulatable, visualize_field, time_recorded
from geometry import Strand, Square, Strand3D, ParamSquare
import numpy as np
import transforms3d


transform_matrix = [
    [-0.9999021887779236, 0.004192245192825794, -0.013345719315111637, -0.05379832163453102],
    [-0.013988681137561798, -0.2996590733528137, 0.95394366979599, 3.845470428466797],
    [-4.656612873077393e-10, 0.9540371894836426, 0.29968830943107605, 1.2080823183059692],
    [0.0, 0.0, 0.0, 1.0]
]

transform_matrix_t = [[-9.9990e-01, -1.3989e-02, -4.6566e-10,  0.0000e+00],
        [ 4.1922e-03, -2.9966e-01,  9.5404e-01,  0.0000e+00],
        [-1.3346e-02,  9.5394e-01,  2.9969e-01,  0.0000e+00],
        [-5.3798e-02,  3.8455e+00,  1.2081e+00,  1.0000e+00]]

transform_matrix0 = [[ 1.0000e+00,  6.1232e-17, -1.0606e-16, -4.2423e-16],
        [-1.2246e-16,  5.0000e-01, -8.6603e-01, -3.4641e+00],
        [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]

H = 400
W = 400
camera_angle_x = 0.6911112070083618
focus = .5 * W / np.tan(.5 * camera_angle_x)


torch.set_default_dtype(torch.float32)
device = 'cpu'


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_axis(c2w):
    look = torch.tensor([0, 0, -1])
    up = torch.tensor([0, 1, 0])

    look = torch.sum(look[None, None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    up = torch.sum(up[None, None, :] * c2w[:3, :3], -1)

    up = up.flatten()
    look = look.flatten()

    return up / torch.norm(up), look / torch.norm(look)


class CameraLocater(Simulatable):

    angle = np.pi / 3

    def __init__(self):
        super().__init__()

    def step(self, dt):
        torch.manual_seed(1)
        pos = torch.rand(1000, 3)
        scene_min, scene_max = torch.tensor([-1, -1, -0.], device=device), torch.tensor([1, 0, 1.], device=device)
        pos = pos * (scene_max - scene_min) + scene_min
        pose = torch.tensor(transform_matrix)

        K = [
            [focus, 0, 0],
            [0, focus, 0],
            [0, 0, 1]
        ]
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        render_pose = pose_spherical(self.angle, -30.0, 4.0)
        render_pose = render_poses[0]
        print(render_pose)
        c2w = pose[:3, :4]
        rays_o, rays_d = get_rays(H, W, K, c2w)
        up, look = get_axis(c2w)

        print(pose.transpose(0, 1))
        # pos = rays_o.view(-1, 3)
        # vec = rays_d.view(-1, 3) * 1000
        # visualize_field(pos, vectors=vec)

        pnt = rays_o.view(-1, 3)[0:1]
        print(pnt)
        axis = look.view(-1, 3) * 100
        visualize_field(pnt, vectors=axis)

        print(focus)
        print('[Ratio]', focus / W)


if __name__ == '__main__':
    CameraLocater().run()
