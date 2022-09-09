import torch

means = [ 1.2565, -1.4614,  0.4473]

covs = [[ 1.3224e-07, -2.9220e-07,  1.6656e-07],
[-2.9162e-07, -2.3298e-07, -3.3778e-09],
[ 1.6654e-07, -3.3306e-09,  2.6773e-07]]
"""
[[5.1703e-07, 3.5736e-08, 2.1054e-07],
        [3.5729e-08, 5.9650e-08, 1.1722e-08],
        [2.1054e-07, 1.1726e-08, 2.1758e-07]]
[[ 1.3224e-07, -2.9220e-07,  1.6656e-07],
        [-2.9162e-07, -2.3298e-07, -3.3778e-09],
        [ 1.6654e-07, -3.3306e-09,  2.6773e-07]]
"""

means = [ 29.5804, -34.4060,  10.5317]
covs = [[ 0.5339, -0.6082,  0.1917],
        [-0.6082,  0.6930, -0.2184],
        [ 0.1917, -0.2184,  0.0689]]


# tst_means, tst_cov = torch.tensor([ 29.5804, -34.4060,  10.5317], device=x.device), torch.tensor([[ 0.5339, -0.6082,  0.1917], [-0.6082,  0.6930, -0.2184], [ 0.1917, -0.2184,  0.0689]], device=x.device)


def joint(a, b):
    return a.unsqueeze(-2) * b.unsqueeze(-1)


def f(x):
    nx = torch.norm(x, dim=-1, keepdim=True)
    f = torch.where(nx <= 1, x, (2 - 1 / nx) * (x / nx))
    return f


def df(x):
    nx = torch.norm(x, dim=-1, keepdim=True)
    nx_ = nx.unsqueeze(-1)
    I = torch.eye(x.size(-1), device=x.device)
    dv = I / nx_ - joint(x, x) / nx_ ** 3
    du = x / nx ** 3
    u = 2 - 1 / nx
    v = x / nx
    J = joint(du, v) + u.unsqueeze(-1) * dv
    return torch.where(nx_ <= 1, I.expand(J.shape), J)


def contract_J(x):

    def joint(a, b):
        return a.unsqueeze(-2) * b.unsqueeze(-1)

    nx = torch.norm(x, dim=-1, keepdim=True)
    nx_ = nx.unsqueeze(-1)
    I = torch.eye(x.size(-1), device=x.device)
    dv = I / nx_ - joint(x, x) / nx_ ** 3
    du = x / nx ** 3
    u = 2 - 1 / nx
    v = x / nx
    J = joint(du, v) + u.unsqueeze(-1) * dv
    return torch.where(nx_ <= 1, I.expand(J.shape), J)


def contract(x, std):
    nx = torch.norm(x, dim=-1, keepdim=True)
    assert (nx >= -1e-8).all()
    f = torch.where(nx <= 1, x, (2 - 1 / nx) * (x / nx))
    J = contract_J(x)
    return f, J @ std @ J.transpose(-2, -1)


def nerf_contract_J(x):

    def joint(a, b):
        return a.unsqueeze(-2) * b.unsqueeze(-1)

    nx = torch.norm(x, dim=-1, keepdim=True)
    nx_ = nx.unsqueeze(-1)
    I = torch.eye(x.size(-1), device=x.device)
    dv = I / nx_ - joint(x, x) / nx_ ** 3
    du = x / nx ** 3
    u = 2 - 1 / nx
    v = x / nx
    J = joint(du, v) + u.unsqueeze(-1) * dv
    return torch.where(nx_ <= 1, I.expand(J.shape), J)


def nerf_contract(x, std):
    nx = torch.norm(x, dim=-1, keepdim=True)
    assert (nx >= -1e-8).all()
    f = torch.where(nx <= 1, x, (2 - 1 / nx) * (x / nx))
    J = nerf_contract_J(x)
    return f, J @ std @ J.transpose(-2, -1)


def tst():
    m = torch.tensor(means)
    c = torch.tensor(covs)

    x, v = nerf_contract(m, c)
    print(v)

    torch.set_default_dtype(torch.double)

    m = torch.tensor(means, device='cuda')
    c = torch.tensor(covs, device='cuda')

    x, v = nerf_contract(m, c)
    print(v)


if __name__ == '__main__':
    tst()


