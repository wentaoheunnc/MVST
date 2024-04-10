import torch
import torch.nn as nn
import torch.nn.functional as fun
from collections import OrderedDict


class FuseMLP(nn.Module):
    def __init__(self, num_features, num_classes=4, representation_size=1024, embed_dim=768):
        super(FuseMLP, self).__init__()

        self.num_classes = num_classes
        W1 = torch.empty(representation_size, representation_size, requires_grad=True)
        W2 = torch.empty(representation_size, representation_size, requires_grad=True)
        W3 = torch.empty(representation_size, representation_size, requires_grad=True)
        W4 = torch.empty(representation_size, representation_size, requires_grad=True)
        W5 = torch.empty(representation_size, representation_size, requires_grad=True)
        nn.init.kaiming_normal_(W1)
        nn.init.kaiming_normal_(W2)
        nn.init.kaiming_normal_(W3)
        nn.init.kaiming_normal_(W4)
        nn.init.kaiming_normal_(W5)
        self.W1 = nn.Parameter(W1)
        self.W2 = nn.Parameter(W2)
        self.W3 = nn.Parameter(W3)
        self.W4 = nn.Parameter(W4)
        self.W5 = nn.Parameter(W5)
        self.fc = nn.Linear(representation_size, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, x3, x4, x5):
        G1 = torch.sigmoid(x1 @ self.W1)
        G2 = torch.sigmoid(x2 @ self.W2)
        G3 = torch.sigmoid(x3 @ self.W3)
        G4 = torch.sigmoid(x4 @ self.W4)
        G5 = torch.sigmoid(x5 @ self.W5)
        F = G1 * x1 + G2 * x2 + G3 * x3 + G4 * x4 + G5 * x5
        output = self.fc(F)

        return output


def new_fuse(num_features: int = 5, num_classes: int = 4, has_logits: bool = True):
    model = FuseMLP(num_features=num_features,
                    num_classes=num_classes,
                    representation_size=768 if has_logits else None,
                    embed_dim=768)
    return model
