import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperPairModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dims=None, dropout_prob=0.3, use_bn=True, bn_momentum=0.9):
        if hidden_dims is None:
            hidden_dims = [2048, 1024]
        super(PaperPairModel, self).__init__()
        input_dim = embedding_dim * 2 + 2
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dims[0], momentum=bn_momentum)) #type: ignore
        layers.extend([nn.Tanh(), nn.Dropout(dropout_prob)]) #type: ignore

        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dims[i], momentum=bn_momentum))#type: ignore
            layers.extend([nn.Tanh(), nn.Dropout(dropout_prob)])#type: ignore

        layers.extend([nn.Linear(hidden_dims[-1], 1), nn.Sigmoid()])#type: ignore
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x1, x2, is_after):
        cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-8).unsqueeze(1)
        x_diff = torch.abs(x1 - x2)
        x_mul = x1 * x2
        # x = torch.cat((x_diff, x_mul, cosine_sim), dim=1)
        x = torch.cat((x_diff, x_mul, is_after.unsqueeze(1), cosine_sim), dim=1)

        x = self.fc_layers(x)
        return x.squeeze()
