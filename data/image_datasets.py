import functools
import logging
import bisect
import os
import torch.utils.data as data
import cv2
import numpy as np
import glob
from config import Configurable, State
import math


class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

    def __init__(self, dataset_name, is_training=True, debug=False, data_dir=None, gt_dir=None, processes=[]):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.processes = processes
        self.is_training = is_training
        self.debug = debug
        self.gt_dir = gt_dir

        if self.gt_dir is not None:
            if self.is_training:
                self.data_list = self.gt_dir + 'train_list.txt'
            else:
                self.data_list = self.gt_dir + 'test_list.txt'
        if self.is_training:
            self.data_dir = self.data_dir + 'train/'
        else:
            self.data_dir = self.data_dir + 'test/'


        self.image_paths = []
        self.gt_paths = []
        self.fids = []
        self.targets = []
        self.num_samples = 0
        self.get_all_samples()

    def get_all_samples(self):
        if 'TD500' in self.data_dir:
            path = self.data_dir
            files = os.listdir(path)
            for file in files:
                fid, ext = file.split('.')
                if ext == 'JPG':
                    img_path = os.path.join(path, file)
                    gt_path = os.path.join(path, fid + '.gt')
                    self.image_paths.append(img_path)
                    self.gt_paths.append(gt_path)
                    self.fids.append(fid)
            self.targets = []
            for path in self.gt_paths:
                polys = self.get_msra_ann(path)
                self.targets.append(polys)
        else:
            with open(self.data_list, 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path = [self.data_dir + 'Images/Train/' + timg.strip() for timg in image_list]
                gt_path = [self.gt_dir + 'train_gts/' + timg.strip() + '.txt' for timg in image_list]
            else:
                image_path = [self.data_dir + 'Images/Test/' + timg.strip() for timg in image_list]
                print(self.data_dir)
                if 'TD500' in self.data_list or 'total_text' in self.data_list:
                    gt_path = [self.gt_dir + 'test_gts/' + timg.strip() + '.txt' for timg in image_list]
                else:
                    gt_path = [self.gt_dir + 'test_gts/' + timg.strip().split('.')[0] + '.txt' for
                               timg in image_list]
            self.image_paths += image_path
            self.gt_paths += gt_path
            self.targets = self.load_ann()
        self.num_samples = len(self.image_paths)
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                if 'TD' in self.data_dir[0] and label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                if 'icdar' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                else:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res


    def get_msra_ann(self, gt_path):
        bboxes = []
        reader = open(gt_path, 'r').readlines()
        for line in reader:
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')
            line = line.replace('\n', '')
            gt = line.split(' ')

            w_ = np.float(gt[4])
            h_ = np.float(gt[5])
            x1 = np.float(gt[2]) + w_ / 2.0
            y1 = np.float(gt[3]) + h_ / 2.0
            theta = np.float(gt[6]) / math.pi * 180

            bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
            bboxes.append({'poly': bbox, 'text': '#', 'ignore': False})

        return np.array(bboxes)

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)
