import numpy as np
import torch

from texture.dev.texture_model import NeuSTextureModel, NeuSTextureRenderer, erode_image
from cluster.match import FocusSampler, SynDataset


data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\hotdog", 4)
sampler = FocusSampler(data)

ntm = NeuSTextureModel("./cache/hotdog_mc.ply", [-1.5, 1.5])
vert = np.array(ntm.mesh.vertices * 0.5)
sample, gt = sampler.scatter_sample(torch.tensor(vert).cuda().float())
color = gt["rgb"]

# color = (color * 255).long()
# max_n = []
# for i in range(255):
#     max_n.append((color == i).sum(0))
# max_n = torch.stack(max_n, 0)
# _, idx = torch.max(max_n, 0)
# print(idx.shape)
# color = idx / 255.0

color = color.mean(0)

color = color.cpu().numpy() * 1.5
color = np.clip(color, 0.01, 0.99)

ntm.save_uint8("vote", color * 255)
erode_image(ntm.get_cache_path("vote"))
ntr = NeuSTextureRenderer()
ntr.show_model("./cache/hotdog_mc.obj", ntm.get_cache_path("vote"))

if __name__ == '__main__':
    pass
