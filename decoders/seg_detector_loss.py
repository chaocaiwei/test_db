import torch
from torch import nn
from decoders.loss import DiceLoss
from decoders.loss import MaskL1Loss
from decoders.loss import BalanceCrossEntropyLoss


class L1BalanceCELoss(nn.Module):

    # α and β are set to 1.0 and 10 respectively
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):

        # loss for the probability map
        prob_loss = self.bce_loss(pred['prob'], batch['gt'], batch['mask'])
        metrics = dict(prob_loss=prob_loss)
        if 'thresh' in pred:
            # the loss for the threshold map
            thresh_loss, l1_metric = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
            # the loss for the binary map
            binary_loss = self.dice_loss(pred['binary'], batch['gt'], batch['mask'])
            metrics['binary_loss'] = binary_loss
            metrics['thresh_loss'] = thresh_loss
            loss = binary_loss + thresh_loss * self.l1_scale + self.bce_scale * prob_loss
            metrics.update(**l1_metric)
        else:
            loss = prob_loss
        return loss, metrics

    def __call__(self, pred, batch):
        return self.forward(pred, batch)

    # prob---binary    thresh---thresh   binary---thresh_binary