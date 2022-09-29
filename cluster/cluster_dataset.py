import os

import torch
from math import sqrt
import winnt
from gmm import GaussianMixture
from torch_kmeans import KMeans
from torchvision.io import read_image, write_png
import glob
import imageio
import numpy as np


def load_rgb(path):
    img = imageio.imread(path)[:, :, :3]
    img = np.float32(img)
    if not path.endswith('.exr'):
        img = img / 255.
        img = np.power(img, 2.2)

    return img


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_dir):
        self.image_dir = image_dir
        image_paths = glob.glob(image_dir)
        image_paths = list(set(image_paths))
        image_paths = list(sorted(image_paths))

        self.rgb_images = []
        self.image_paths = []

        for path in image_paths:
            abs_path = os.path.join(image_dir, path)
            self.image_paths.append(abs_path)
            rgb = load_rgb(abs_path).reshape(-1, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        return self.rgb_images[idx]


def transfer(image_dir, out_dir, n_components=16):
    dataset = Dataset(image_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    model = GaussianMixture(n_components, 3)
    model.cuda()

    for data_index, images in enumerate(dataloader):
        model.fit(images)

    # y = model.predict(data)


if __name__ == '__main__':
    transfer()

