import torch.nn as nn


class MLPPredictor(nn.Module):
    def __init__(self, device, ckpt=None):
        super(MLPPredictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.to(device)
        if ckpt is not None:
            self.load_state_dict(ckpt)

    def forward(self, x_edge):
        return self.pred(x_edge)
