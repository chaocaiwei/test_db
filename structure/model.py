import os

import torch
import torch.nn as nn

import backbones.resnet
import backbones.res50_se_net
from decoders.seg_detector_asff import SegDetectorAsff
from decoders.seg_detector import SegDetector


class BasicModel(nn.Module):
    def __init__(self, backbone, decoder, backbone_args, decoder_args):
        nn.Module.__init__(self)
        if 'se' in backbone:
            self.backbone = getattr(backbones.res50_se_net, backbone)()
        else:
            self.backbone = getattr(backbones.resnet, backbone)()
        if decoder == 'SegDetectorAsff':
            self.decoder = SegDetectorAsff(**decoder_args)
        else:
            self.decoder = SegDetector(**decoder_args)

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)


class SegDetectorModel(nn.Module):
    def __init__(self, backbone, decoder,  decoder_args, device, backbone_args=None, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import L1BalanceCELoss

        # TODO 加入其他loss算法
        self.criterion = L1BalanceCELoss()

        self.model = BasicModel(backbone, decoder, backbone_args, decoder_args)

        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data)

        if self.training and batch is dict:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            return pred
        return pred

