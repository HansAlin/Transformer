import torch
import torch.nn as nn

class HingeLoss(nn.Module):
    def __init__(self, device, hinge_type='hinge'):
        super(HingeLoss, self).__init__()
        self.device = device
        self.hinge_type = hinge_type

        if self.hinge_type == 'hinge':
            self.loss = self.hinge_loss

    def hinge_loss(self, y_pred, y_true):
        hinge_lables = 2 * y_true - 1
        pre_loss = 1 - hinge_lables * y_pred
        max_loss = torch.max(torch.tensor(0, dtype=torch.float32, device=self.device), pre_loss)
        loss = torch.mean(max_loss)
        return loss

    def forward(self, y_pred, y_true):
        loss = self.loss(y_pred, y_true)
        return loss