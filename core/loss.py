import torch
import torch.nn as nn


class CombinedMSEMAE(nn.Module):
    def __init__(self):
        super(CombinedMSEMAE, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, output, target):
        comb = 0.5 * self.mse(output, target) + 0.5 * self.mae(output, target)
        return comb


class CharbonnierL1Loss(torch.nn.modules.loss._Loss):
    r"""Charbonnier L1 Loss from Equation 14:

    Wang, Zhihao, Jian Chen, and Steven CH Hoi.
    "Deep learning for image super-resolution: A survey."
    IEEE PAMI 43.10 (2020): 3365-3387.

    """

    def __init__(self, epsilon=1e-2, reduction="mean"):
        super(CharbonnierL1Loss, self).__init__(reduction=reduction)
        self.epsilon = epsilon

    def forward(self, output, target):
        # get batch size
        batch_size = output.shape[0]

        L = torch.mean(
            torch.sqrt(
                (output.reshape(batch_size, -1) - target.reshape(batch_size, -1)) ** 2
                + self.epsilon**2
            ),
            dim=1,
        )

        # apply selected reduction method
        if self.reduction == "mean":
            return L.mean()
        elif self.reduction == "sum":
            return L.sum()
        else:  # none
            return L


def get_loss(name):
    lossname_to_func = {
        "XENT": nn.CrossEntropyLoss(),
        "L2": nn.MSELoss(),
        "L1": nn.L1Loss(),
        "SMOOTHL1": nn.SmoothL1Loss(),
        "CHARBL1": CharbonnierL1Loss(),
        # our custom loss
        "COMB": CombinedMSEMAE(),
    }
    return lossname_to_func[name]
