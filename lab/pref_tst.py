import torch
import torch.nn as nn
import torch.nn.functional as F
from lab.pref_encoder import PREF


class NeuralField(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PREF(linear_freqs=[128] * 3, reduced_freqs=[1] * 3, feature_dim=16)

        input_dim = self.encoder.output_dim
        hidden_dim = 64
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x


if __name__ == '__main__':
    nf = NeuralField()
    nf.cuda()
    x = torch.rand(100, 3).cuda()
    y = nf(x)
    print(y.shape)
