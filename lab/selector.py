import torch

t = torch.tensor([[[1, 11], [2, 22], [3, 33]], [[4, 44], [5, 55], [6, 66]]])
print(t.shape)
res = torch.gather(t, 0, torch.tensor([[[0, 1], [0, 0]], [[1, 0], [1, 0]]]))

print(res)

if __name__ == '__main__':
    pass
