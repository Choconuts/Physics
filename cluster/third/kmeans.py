import torch
from math import sqrt
from gmm import GaussianMixture
from torch_kmeans import KMeans
from torchvision.io import read_image, write_png


def kmeans():
    img = read_image("../../lab/lego.png").permute(1, 2, 0)[..., :3] / 255.0
    print(img.shape)

    model = KMeans(n_clusters=32)

    result = model(img)
    mask = (result.labels * 255.0 / 4).type(torch.uint8)[None].expand(3, -1, -1)
    write_png(mask, "../label.png")


def main():

    # generate some data points ..
    data = read_image("../dh.jpg").permute(1, 2, 0)[..., :3]
    H, W, _ = data.shape
    data = (data / 255.0).view(-1, 3)
    n, d = data.shape

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 8
    model = GaussianMixture(n_components, d)
    # model.cuda()

    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)

    mask = (y * 255 / n_components).type(torch.uint8).view(1, H, W).expand(3, -1, -1)
    write_png(mask, "../label.png")


def huge_main():

    # generate some data points ..
    data = (read_image("../../lab/lego.png").permute(1, 2, 0)[..., :3] / 255.0).view(-1, 3).repeat(100, 1)
    n, d = data.shape

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 8
    model = GaussianMixture(n_components, d)

    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)

    mask = (y[:800*800] * 255 / n_components).type(torch.uint8).view(1, 800, 800).expand(3, -1, -1)
    write_png(mask, "../label.png")


def cuda_main():

    # generate some data points ..
    data = (read_image("../../lab/lego.png").permute(1, 2, 0)[..., :3] / 255.0).view(-1, 3)
    n, d = data.shape

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 8
    model = GaussianMixture(n_components, d)
    model.cuda()

    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)

    mask = (y * 255 / n_components).type(torch.uint8).view(1, 800, 800).expand(3, -1, -1)
    write_png(mask, "../label.png")


if __name__ == '__main__':
    main()

