import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import densenet121

class DenseNet121(nn.Module):
    def __init__(self, args):
        super(DenseNet121, self).__init__()

        self.densenet121 = densenet121(pretrained=True)
        n_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(n_features, args.n_classes), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x