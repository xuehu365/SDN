import torch
import torch.nn as nn

class CatDomain(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(CatDomain, self).__init__()

        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        y = self.layer3(x)

        return y

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]
