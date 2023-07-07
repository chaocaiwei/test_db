import argparse
import os
import torch
import numpy as np
from structure.model import SegDetectorModel
import onnx

class ONNXConverter:
    def __init__(self, model, model_path, output_dir, name):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.model_path = model_path

        self.model = model

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        self.output_path = os.path.join(output_dir,  name + '.onnx')

    def init_torch_tensor(self):
        # Use gpu or not
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        states = torch.load(path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def convert(self):
        self.init_torch_tensor()
        model = self.model
        self.resume(model, self.model_path)
        model.eval()

        img = np.random.randint(0, 255, size=(960, 960, 3), dtype=np.uint8)
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5  # torch style norm
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        with torch.no_grad():
            img = img.to(self.device)
            torch.onnx.export(model.model.module, img, self.output_path, input_names=['input'],
                              output_names=['output'], dynamic_axes=dynamic_axes, keep_initializers_as_inputs=False,
                              verbose=False, opset_version=12)

    def load(self, path):
        model = onnx.load(path)
        return model


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == '__main__':
    backbone = 'resnet50_se_net'
    decoder = 'SegDetectorAsff'
    decoder_args = {'in_channels': [256, 512, 1024, 2048]}
    device = try_gpu()
    model = SegDetectorModel(backbone, decoder, decoder_args, device)
    model_path = '/Users/myself/Desktop/model/td500_7_2'
    result_dir = '/Users/myself/Desktop/outputs/onnx/'
    cvt = ONNXConverter(model=model, model_path=model_path, output_dir=result_dir, name='td500')
    #cvt.convert()

    cvt.load(os.path.join(result_dir, 'td500.onnx'))