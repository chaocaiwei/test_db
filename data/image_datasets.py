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

    def __init__(self, dataset_name, is_training=True, debug=False, data_dir=None, processes=[], cmd={}):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.processes = processes
        self.is_training = is_training
        self.debug = debug

        if 'data_dir' in cmd:
            data_dir = cmd['data_dir']
            data_list = data_dir + '/train_list.txt' if self.is_training else data_dir + '/test_list.txt'
            self.data_dir = [data_dir]
            self.data_list = [data_list]

        self.image_paths = []
        self.gt_paths = []
        self.fids = []
        self.targets = []
        self.num_samples = 0
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path=[self.data_dir[i]+'/train_images/'+timg.strip() for timg in image_list]
                gt_path=[self.data_dir[i]+'/train_gts/'+timg.strip()+'.txt' for timg in image_list]
            else:
                image_path=[self.data_dir[i]+'/test_images/'+timg.strip() for timg in image_list]
                print(self.data_dir[i])
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    gt_path=[self.data_dir[i]+'/test_gts/'+timg.strip()+'.txt' for timg in image_list]
                else:
                    gt_path=[self.data_dir[i]+'/test_gts/'+'gt_'+timg.strip().split('.')[0]+'.txt' for timg in image_list]
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
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

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print('image is none ', image_path)
            return {'image': None, 'filename': image_path}
        img = img.astype('float32')
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
