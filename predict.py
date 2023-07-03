import torch
from structure.model import SegDetectorModel
import cv2
from structure.represent import SegDetectorRepresenter
from structure.measurer import QuadMeasurer
from structure.visualizer import SegDetectorVisualizer
from data.processes.augment_data import AugmentDetectionData
from data.processes.data_process import *
import math
import os
from data.data_loader import DataLoader
import imgaug.augmenters as iaa
from backbones.res50_se_net import resnet50_se_net
from decoders.seg_detector_asff import SegDetectorAsff
from decoders.seg_detector_loss import L1BalanceCELoss
from d2l import torch as d2l




def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def create_model(model_path):
    backbone = 'resnet50_se_net'
    decoder = 'SegDetectorAsff'
    decoder_args = {'in_channels': [256, 512, 1024, 2048]}
    device = try_gpu()
    model = SegDetectorModel(backbone, decoder, decoder_args, device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return model


def resize_image(image, height):
    origin_width, origin_height = image.shape[0], image.shape[1]
    width = origin_width * height / origin_height
    N = math.ceil(width / 32)
    width = N * 32
    image = cv2.resize(image, (width, height))
    return image


def norm_image(image):
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
    image = np.array(image, dtype=float)
    image -= RGB_MEAN
    image /= 255.
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    return image


def processes_image(img):
    img = resize_image(img, 736)
    img = norm_image(img)
    return img


def format_output(output, filename, result_dir, polygon=True, box_thresh=0.3):
    batch_boxes, batch_scores = output
    boxes = batch_boxes[0]
    scores = batch_scores[0]
    result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
    result_file_path = os.path.join(result_dir, result_file_name)
    if polygon:
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
                if score < box_thresh:
                    continue
                box = boxes[i, :, :].reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def eval_model(name, model, batch, filename, is_output_polygon=False, thresh=0.35, box_thresh=0.7):
    result_dir = os.path.join('eval_result/', name)

    representer = SegDetectorRepresenter(max_candidates=1000, thresh=thresh, box_thresh=box_thresh)
    measurer = QuadMeasurer()
    visualizer = SegDetectorVisualizer(eager_show=False)

    img = batch['image']
    if len(img.shape) == 3:
        batch['image'] = img.reshape((1, *img.shape))
        batch['shape'] = batch['shape'].reshape((1, *batch['shape'].shape))

    model.eval()
    with torch.no_grad():
        pred = model.forward(batch)
        output = representer.represent(batch, pred, is_output_polygon=is_output_polygon)
        print('output', output)
        if not os.path.isdir('eval_result'):
            os.mkdir('eval_result')
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        format_output(output, filename, result_dir, polygon=is_output_polygon, box_thresh=box_thresh)

        # batch = {'image': img, 'polygons': [], 'ignore_tags': [], 'filename': filename}
        raw_metric = measurer.validate_measure(batch, output, is_output_polygon=is_output_polygon,
                                               box_thresh=box_thresh)
        vis_image = visualizer.visualize(batch, output, pred)
        metrics = measurer.gather_measure([raw_metric], None)
        for key, metric in metrics.items():
            print('%s : %f (%d)' % (key, metric.avg, metric.count))

        # binary = pred['binary'].detach().numpy()
        # binary_b = (binary > thresh).astype(np.float64)
        # prob = pred['prob'].detach().numpy()
        # prob_b = (prob > thresh).astype(np.float64)
        # cv2.imshow(name + '.prob', prob[0][0])
        # cv2.imshow(name + '.prob-b', prob_b[0][0])
        # cv2.imshow(name + '.thresh', pred['thresh'][0][0].detach().numpy())
        # cv2.imshow(name + '.binary', pred['binary'][0][0].detach().numpy())
        # cv2.imshow(name + '.binary-b', binary_b[0][0])


if __name__ == '__main__':

    image_path = '/Users/myself/Desktop/datasets/total_text/Images/Train/img17.jpg'
    is_output_polygon = True
    thresh = 0.3
    box_thresh = 0.4

    org_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow('org image', org_img)

    model_path_1 = '/Users/myself/Desktop/model/td500_adw_onecircle'
    model_path_2 = '/Users/myself/Desktop/model/td500_sgd'
    model_origin = '/Users/myself/Desktop/model/totaltext_sgd'
    model_1 = create_model(model_path_1)
    model_2 = create_model(model_path_2)
    model_origin = create_model(model_origin)

    data_dir = '/Users/myself/Desktop/datasets/MSRA-TD500/'
    processes = [{'class': 'AugmentDetectionData', 'augmenter_args': [['Resize', {'width': 736, 'height': 736}]],
                  'only_resize': True, 'keep_ratio': True},
                 {'class': 'MakeICDARData'}, {'class': 'MakeSegDetectionData'}, {'class': 'NormalizeImage'}]
    dataset = {'dataset_name': 'td500', 'data_dir': data_dir, 'processes':  processes}
    loader = DataLoader(dataset, batch_size=1, num_workers=1, is_training=True, shuffle=False)
    i = 0
    for batch in loader:
        if i < 5:
            continue
        i += 1
        eval_model('td500_adw_onecircle', model_origin, batch, image_path, is_output_polygon=is_output_polygon,
                   box_thresh=box_thresh, thresh=thresh)
        break



    #img = processes_image(org_img)
    #eval_model('td500_adw_onecircle', model_1, img, image_path, is_output_polygon=is_output_polygon, box_thresh=box_thresh, thresh=thresh)
    #eval_model('td500_sgd', model_2, img, image_path, is_output_polygon=is_output_polygon, box_thresh=box_thresh)

    # cv2.waitKey()

