import torch
from structure.model import SegDetectorModel
import cv2
from structure.represent import SegDetectorRepresenter
from structure.measurer import QuadMeasurer
from structure.visualizer import SegDetectorVisualizer
from data.processes.augment_data import AugmentDetectionData
from data.processes.data_process import *
import imgaug.augmenters as iaa
from backbones.res50_se_net import resnet50_se_net
from decoders.seg_detector_asff import SegDetectorAsff
from decoders.seg_detector_loss import L1BalanceCELoss
from d2l import torch as d2l

model_path = '/Users/myself/Desktop/model/td500_adw_onecircle'


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == '__main__':

    backbone = 'resnet50_se_net'
    decoder = 'SegDetectorAsff'
    decoder_args = {'in_channels': [256, 512, 1024, 2048]}
    device = try_gpu()
    model = SegDetectorModel(backbone, decoder, decoder_args, device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    image_path = 'datasets/MSRA-TD500/train/IMG_0451.JPG'
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
    batch = {'image': img, 'lines': []}
    arg = {'only_resize': True, 'keep_ratio': True, 'augmenter_args': [['Resize', {'width': 736, 'height': 736}]]}
    aug = AugmentDetectionData(arg)
    # pc1 = MakeICDARData()
    # pc2 = MakeSegDetectionData()
    pc3 = NormalizeImage()
    batch = pc3.process(aug.process(batch))

    image = batch['image']
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    batch['image'] = image

    model.eval()
    pred = model.forward(batch)
    representer = SegDetectorRepresenter(max_candidates=1000)
    measurer = QuadMeasurer()
    visualizer = SegDetectorVisualizer()
    output = representer.represent(batch, pred)
    # raw_metric = measurer.validate_measure(batch, output)

    # vis_image = visualizer.visualize(batch, output, pred)

    print(output)
    binary = pred['binary'].detach().numpy()
    binary_b = (binary > 0.5).astype(np.float)
    cv2.imshow('image', image[0][0].detach().numpy())
    cv2.imshow('prob', pred['prob'][0][0].detach().numpy())
    cv2.imshow('thresh', pred['thresh'][0][0].detach().numpy())
    cv2.imshow('binary', pred['binary'][0][0].detach().numpy())
    cv2.imshow('binary-b', binary_b[0][0])
    cv2.waitKey()

