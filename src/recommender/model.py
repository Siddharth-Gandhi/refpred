import torch
import torch.nn as nn


class PaperPairModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super(PaperPairModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()