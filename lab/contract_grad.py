import torch
from torch.autograd import grad


def f(x):
    nx = torch.norm(x, dim=-1, keepdim=True)
    f = torch.where(nx <= 1, x, (2 - 1 / nx) * (x / nx))
    return f


def df(x):
    nx = torch.norm(x, dim=-1, keepdim=True)
    xT = x.unsqueeze(-2)
    x = x.unsqueeze(-1)
    return (x / nx ** 3) * (xT / nx + (2 - 1 / nx)) * (torch.eye(x.size(-1)) / nx - x * xT / nx ** 3)


def u(x):
    return 2 - 1 / torch.norm(x, dim=-1, keepdim=True)


def v(x):
    return x / torch.norm(x, dim=-1, keepdim=True)


def df_gt(x):
    y = f(x)
    grads = [grad(y[i], x, retain_graph=True)[0] for i, _ in enumerate(y)]
    J = torch.stack(grads)
    return J


def du_gt(x):
    y = u(x)
    return grad(y, x, retain_graph=True)[0]


def dv_gt(x):
    y = v(x)
    grads = [grad(y[i], x, retain_graph=True)[0] for i, _ in enumerate(y)]
    J = torch.stack(grads)
    return J


def df_comp(x):
    return du_gt(x).unsqueeze(-2) * v(x).unsqueeze(-1) + u(x) * dv_gt(x)


def du(x):
    nx = torch.norm(x, dim=-1, keepdim=True)
    return x / nx ** 3


def dv(x):
    nx = torch.norm(x, dim=-1, keepdim=True)
    p2 = x.unsqueeze(-2) * (x).unsqueeze(-1) / nx ** 3
    return torch.eye(x.size(-1)) / nx + -p2


def joint(a, b):
    return a.unsqueeze(-2) * b.unsqueeze(-1)


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


if __name__ == '__main__':
    # x = torch.ones(3)
    # x.requires_grad_(True)
    # res = f(x)
    # print(df(x))
    # print(df_gt(x))

    x = torch.rand(3) * 3
    x.requires_grad_(True)
    print(df(x))
    print(df_gt(x))
