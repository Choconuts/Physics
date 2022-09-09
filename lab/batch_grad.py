import torch

torch.random.manual_seed(0)
x = torch.rand(10, 3).requires_grad_(True)
y = x ** 2
g = torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)
g2 = torch.autograd.grad(y.sum(), x, retain_graph=True)

with torch.no_grad():
    x.requires_grad_(True)
    print(x.requires_grad)
    print(torch.is_grad_enabled())
    with torch.enable_grad():
        x.requires_grad_(True)
        print(x.requires_grad)
        print(torch.is_grad_enabled())

if __name__ == '__main__':
    pass
