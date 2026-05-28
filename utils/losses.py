import torch
import torch.nn as nn


# Dice Loss
class DiceLoss(nn.Module):

    def __init__(self, smooth=1):

        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, preds, targets):

        preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()

        dice = (
            2. * intersection + self.smooth
        ) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


# Combined BCE + Dice Loss
class BCEDiceLoss(nn.Module):

    def __init__(self):

        super(BCEDiceLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(preds, targets)

        dice_loss = self.dice(preds, targets)

        total_loss = bce_loss + dice_loss

        return total_loss