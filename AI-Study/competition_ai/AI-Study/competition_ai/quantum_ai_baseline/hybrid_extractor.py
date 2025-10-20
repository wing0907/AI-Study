# hybrid_extractor.py
import torch
from torch import nn
from quantum_model import qnn_torch
from classical_model import ClassicalCNN

class HybridExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(28 * 28, 4)
        self.qnn   = qnn_torch
        self.cnn   = ClassicalCNN()

    def forward(self, x):
        b    = x.size(0)
        flat = x.view(b, -1)
        emb  = self.embed(flat)
        x_q  = self.qnn(emb)
        x_c  = self.cnn(x)
        return torch.cat([x_q, x_c], dim=1)