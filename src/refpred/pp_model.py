import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperPairModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=2048, num_hidden_layers=2, dropout_prob=0.3, use_bn=True, bn_momentum=0.9):
        super(PaperPairModel, self).__init__()

        layers = [nn.Linear(embedding_dim * 2 + 2, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)) # type: ignore
        layers.extend([nn.LeakyReLU(), nn.Dropout(dropout_prob)]) # type: ignore

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))  # type: ignore
            layers.extend([nn.LeakyReLU(), nn.Dropout(dropout_prob)])  # type: ignore

        layers.extend([nn.Linear(hidden_dim, 1), nn.Sigmoid()]) # type: ignore
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x1, x2, is_after):
        cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-8).unsqueeze(1)
        x_diff = torch.abs(x1 - x2)
        x_mul = x1 * x2
        x = torch.cat((x_diff, x_mul, is_after.unsqueeze(1), cosine_sim), dim=1)
        x = self.fc_layers(x)
        return x.squeeze()
