import torch
import torch.nn as nn

class HingeLoss(nn.Module):
    def __init__(self, hinge_type='hinge'):
        super(HingeLoss, self).__init__()
        self.hinge_type = hinge_type

        if self.hinge_type == 'hinge':
            self.loss = self.hinge_mean
        elif self.hinge_type == 'hinge_max':
            self.loss = self.hinge_max
        else:
            raise ValueError('Unknown hinge type')    

    def hinge_mean(self, y_pred, y_true):
        hinge_lables = 2 * y_true - 1
        pre_loss = 1 - hinge_lables * y_pred
        max_loss = torch.max(torch.tensor(0, dtype=torch.float32), pre_loss)
        loss = torch.mean(max_loss)
        return loss

    def hinge_max(self, y_pred, y_true):
        hinge_lables = 2 * y_true - 1
        pre_loss = 1 - hinge_lables * y_pred
        max_loss = torch.max(torch.tensor(0, dtype=torch.float32), pre_loss)
        loss = torch.max(max_loss)
        return loss

    def forward(self, y_pred, y_true):
        loss = self.loss(y_pred, y_true)
        return loss