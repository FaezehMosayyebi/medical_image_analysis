#     Description: The Soft Dice Loss is a variant of the Dice Loss function introduced in   
#     the paper 'https://arxiv.org/abs/1606.04797'. The used loss function combines the      
#     advantages of CrossEntropyLoss and DiceLoss to effectively calculate the overall loss  
#     during the process.                                                                    


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, smooth=1.0):
        super(SoftDiceLoss, self).__init__()

        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y):
        # x is network out_put and y is our seg
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))  # makes axes = [b,x,y,z]
        else:
            axes = list(range(2, len(shp_x)))  # axes = [x,y,z]

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            y_onehot = y

        intersect = x * y_onehot  # finding true positives
        # values in the denominator get smoothed
        denominator = x**2 + y_onehot**2

        # claculating the number of ones intersect and denominator
        for ax in sorted(axes, reverse=True):
            intersect = intersect.sum(int(ax)) + self.smooth
            denominator = denominator.sum(int(ax)) + self.smooth

        dc = 2 * intersect / denominator
        dc = dc.mean()

        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(
        self, soft_dice_kwargs, ce_kwargs, aggregate="sum", weight_ce=1, weight_dice=1
    ):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.

        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate

        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=None, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        """
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        new_loss = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                new_loss += weights[i] * self.loss(x[i], y[i])
        return new_loss
