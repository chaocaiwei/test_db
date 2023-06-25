import imgaug
import numpy as np
import imgaug.augmenters as iaa
from config import State
from .data_process import DataProcess
import cv2
import math


class AugmenterBuilder(object):
    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None:
            return None
        elif isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(**{k: self.to_tuple_if_list(v) for k, v in args.items() if not k == 'cls'})
            else:
                return {key: self.build(value, root=False) for key, value in args.items()}
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class AugmentDetectionData:

    def __init__(self, arg):
        self.augmenter_args = arg['augmenter_args']
        self.keep_ratio = arg['keep_ratio']
        self.only_resize = arg['only_resize']
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True
        else:
            data['is_training'] = False
        return data
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly




