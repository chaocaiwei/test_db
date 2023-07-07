import torch
from structure.model import SegDetectorModel
import cv2
from structure.represent import SegDetectorRepresenter
from structure.measurer import QuadMeasurer
from structure.visualizer import SegDetectorVisualizer
from data.processes.data_process import *
import math
import os


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Predict:

    def __init__(self, model_path, result_dir='../outputs/', image_short_side=736, thresh=0.3, box_thresh=0.6, is_output_polygon=True):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.model_path = model_path
        self.image_short_side = image_short_side
        self.result_dir = result_dir
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.is_output_polygon = is_output_polygon

        self.representer = SegDetectorRepresenter(max_candidates=1000, thresh=thresh, box_thresh=box_thresh)
        self.measurer = QuadMeasurer()
        self.visualizer = SegDetectorVisualizer(eager_show=False)


    def init_torch_tensor(self):
         # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(model_path):
        backbone = 'resnet50_se_net'
        decoder = 'SegDetectorAsff'
        decoder_args = {'in_channels': [256, 512, 1024, 2048]}
        device = try_gpu()
        model = SegDetectorModel(backbone, decoder, decoder_args, device)

        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, image):
        height, width, _ = image.shape
        if height < width:
            new_height = self.image_short_side
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.image_short_side
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(image, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.result_dir, result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.is_output_polygon:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.box_thresh:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def predict(self, image_path, visualize=True):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        img, original_shape = self.load_image(image_path)
        batch = {
            'filename': [image_path],
            'shape': [original_shape],
            'image': img
        }
        with torch.no_grad():
            pred = model.forward(batch)
            output = self.representer.represent(batch, pred, is_output_polygon=self.is_output_polygon)
            if not os.path.isdir(self.result_dir):
                os.mkdir(self.result_dir)
            self.format_output(batch, output)

            if visualize and self.visualizer:
                vis_image = self.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.result_dir, image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)
        return vis_image


def test_pred(is_output_polygon):
    model_path_1 = '/Users/myself/Desktop/model/td500_7_2'
    image_path = '/Users/myself/Desktop/datasets/total_text/Images/Train/img17.jpg'
    result_dir = '/Users/myself/Desktop/outputs/predict/'

    pred = Predict(model_path=model_path_1, result_dir=result_dir, box_thresh=0.5, image_short_side=736, is_output_polygon=is_output_polygon)
    ret = pred.predict(image_path=image_path, visualize=True)


if __name__ == '__main__':
    img_path = '/Users/myself/Desktop/outputs/predict/img17.jpg'
    res_path = '/Users/myself/Desktop/outputs/predict/res_img17.txt'

    test_pred(is_output_polygon=T)
