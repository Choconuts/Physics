import os
import torch
from cluster.third.model.fields import VNeRF


class MyNeRF(torch.nn.Module):

    def __init__(self):
        super(MyNeRF, self).__init__()
        NEUS_LOG_DIR = r"G:\WorkSpace\nerf\logs\dh-vanilla"
        vnerf = VNeRF()
        vnerf.cuda()
        state = torch.load(os.path.join(NEUS_LOG_DIR, "040000.tar"))
        step = state['global_step']
        vnerf.load_state_dict(state['model'])
        self.vnerf = vnerf

    def forward(self, x, dirs):
        x = x * 2.0
        return self.vnerf(x, dirs)

    def density(self, x):
        x = x * 2.0
        return self.vnerf.density(x)


if __name__ == '__main__':
    my_nerf = MyNeRF()
    x = torch.rand(10000, 3).cuda()
    d = torch.ones_like(x)

    rgb, a = my_nerf(x, d)
    print(a.shape, rgb.shape)
    a2 = my_nerf.density(x)
    print(((a - a2).abs() < 0.0001).all())
