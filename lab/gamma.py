import cv2
import numpy as np
import torch
from lab.explicit_sparsity import SDFNetwork
from tqdm import tqdm, trange


class CRF(torch.nn.Module):

    def __init__(self):
        super(CRF, self).__init__()
        self.s = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def f(self, x):
        return 1 / (1 + torch.exp(-self.s * x))

    def inv_f(self, x):
        x = torch.clamp(x, min=1e-4, max=1-1e-4)
        return -self.s * torch.log(1 / x - 1)

    def g(self, x):
        pass

    def inv_g(self, x):
        return torch.exp2(x) * 2.5

    def inv(self, x):
        y = self.inv_g(self.inv_f(x))
        return torch.clamp(y, max=1e5)

    def getter_and_setter(self, i):
        def setter(v):
            tmp = torch.tensor([1.0, 1.0, 1.0])
            for j in range(3):
                tmp[j] = self.s[j]
            tmp[i] = v
            self.s = torch.nn.Parameter(tmp)
            print(self.s)
        return lambda : self.s[i], setter


img = cv2.imread("lego.png") / 255.0
mask = cv2.imread("lego.png", -1)[..., 2] / 255.0
img = cv2.resize(img, (400, 400))
mask = cv2.resize(mask, (400, 400))
img = torch.tensor(img)
mask = torch.tensor(mask) > 0.5


def polar(img):
    img = torch.clamp(img, min=1e-6)
    x, y, z = img[..., 0], img[..., 1], img[..., 2]
    r = torch.norm(img, dim=-1)
    phi = torch.arccos(z / r)
    theta = torch.arcsin(y / r / torch.sin(phi))
    pol = torch.stack([theta, phi, mask * 0.5], -1)

    return pol


class UI:

    def __init__(self):
        def end(): self.running = False

        self.operation = {
            ' ': end
        }
        self.running = True

    def positive_float_value(self, inc, dec, getter, setter, step=0.1):
        def increase():
            v = getter()
            v = v + step
            setter(v)

        def decrease():
            v = getter()
            v = v - step
            v = torch.clamp(v, min=step)
            setter(v)

        self.operation[inc] = increase
        self.operation[dec] = decrease

    def parse(self, k):
        for op in self.operation:
            if k & 0xFF == ord(op):
                self.operation[op]()


crf = CRF()
ui = UI()
ui.positive_float_value('q', 'a', *crf.getter_and_setter(0))
ui.positive_float_value('w', 's', *crf.getter_and_setter(1))
ui.positive_float_value('e', 'd', *crf.getter_and_setter(2))

while ui.running:
    tmp = crf.inv(img)
    p = polar(tmp).detach().cpu().numpy()
    tmp = polar(crf.inv(img))[..., :2]
    if tmp.isnan().any():
        print(tmp.isnan().nonzero())
    hist = torch.histogramdd(tmp.view(-1, 2), (400, 400)).hist
    h = hist.detach().cpu().numpy()
    h = np.stack([h, h, h], -1)
    cv2.imshow("hist", np.concatenate([h, p], 0))
    ui.parse(cv2.waitKey())


class DiffCluster(torch.nn.Module):

    def __init__(self):
        super(DiffCluster, self).__init__()
        self.weight = SDFNetwork(3, 1, 64, 4, (2,))
        self.weight_act = lambda x: torch.nn.functional.softplus(x - 10)

    def sample(self, z, n):
        zs = z[:, None, -1].expand(-1, n, -1)
        noise = torch.randn_like(zs) * 0.2
        return zs + noise

    def forward(self, z):
        zs = self.sample(z, 128)
        w = self.weight_act(self.weight(zs))
        acc = w.sum(-2)
        return (acc - 1.0).abs().mean()


def improve_train():
    torch.manual_seed(3)
    # model = NaiveModel()
    model = DiffCluster()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0005, (0.9, 0.99))
    crf = CRF()

    tmp = polar(crf.inv(img))[..., :2]
    hist = torch.histogramdd(tmp.view(-1, 2), (200, 200)).hist
    cv2.imshow("hist", hist.detach().cpu().numpy())
    cv2.waitKey()

    pbar = trange(30001)
    for i in pbar:
        z, coef, ofs, result = problem.sample_data(5000)
        # index = problem.z_to_id(z).long()
        x, c, acc = model(z, ofs, result, i % 1000 == 200)

        final_result = x * (c + ofs)
        loss = ((final_result - result) ** 2).mean()
        # loss = (index - c) ** 2

        # KL = torch.tensor(0., device=z.device)
        KL = model.density(torch.rand_like(x)).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        total_loss = loss + KL * 0.1 + acc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            x_gt = problem.x[problem.z_to_id(z)][:, 0, :]
            x_diff = (x - x_gt).abs()
            loss = (x * (coef + ofs) - result) ** 2
            loss = loss.mean()

            diff = ((x[..., None, :] - problem.x).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item(), x_diff=x_diff.mean().item())

        if i % 1000 == 0:
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            print(list(map(torch.Tensor.item, x[0])))
            # print((model.field(z)[0] > 1e-3).long())
            pass


if __name__ == '__main__':
    improve_train()
