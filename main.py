# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import shutil
import argparse
import numpy as np
import math
import cv2

def mov_fild(base, training):

    source_path = base + 'train/' if training else base + 'test/'
    i_path = base + 'train_images/' if training else base + 'test_images/'
    gt_path = base + 'train_gts/' if training else base + 'test_gts/'

    if not os.path.isdir(i_path):
        os.makedirs(i_path)
    if not os.path.isdir(gt_path):
        os.makedirs(gt_path)

    files = os.listdir(source_path)
    for file in files:
        fid, ext = file.split('.')
        cur_path = os.path.join(source_path, file)
        if ext == 'JPG':
            shutil.move(cur_path, i_path)
        elif ext == 'gt':
            m_path = gt_path + file
            rename_path = gt_path + file + '.txt'
            shutil.move(cur_path, gt_path)
            os.rename(m_path, rename_path)


def create_list_path(base, training):

    source_path = base + 'train_images/' if training else base + 'test_images/'
    list_path = base + 'train_list.txt' if training else base + 'test_list.txt'

    files = os.listdir(source_path)
    with open(list_path, 'w') as fd:
        for file in files:
            fd.write(file + '\n')


def rename(dir, is_training):
    path = dir + 'train_gts/' if is_training else dir + 'test_gts/'
    files = os.listdir(path)
    for file in files:
        fid = file.split('.')[0]
        cur_path = os.path.join(path, file)
        rename_path = path + fid + '.JPG.txt'
        os.rename(cur_path, rename_path)


def modify_gts(base, is_training):
    path = base + 'train_gts/' if is_training else base + 'test_gts/'
    files = os.listdir(path)
    for file in files:
        cur_path = os.path.join(path, file)
        reader = open(cur_path, 'r').readlines()
        txts = []
        for line in reader:
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')
            line = line.replace('\n', '')
            gt = line.split(' ')

            is_difficult = np.int64(gt[1])
            w_ = np.float64(gt[4])
            h_ = np.float64(gt[5])
            x1 = np.float64(gt[2]) + w_ / 2.0
            y1 = np.float64(gt[3]) + h_ / 2.0
            theta = np.float64(gt[6]) / math.pi * 180

            bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
            points = list(bbox.reshape(-1))
            dst_txt = ','.join([str(i) for i in points]) + ',' + str(is_difficult)
            txts.append(dst_txt)
        with open(cur_path, 'r+') as file:
            file.truncate(0)
            for txt in txts:
                file.write(txt + '\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('base', type=str)
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    base = args['base']

    #mov_fild(base, True)
    #mov_fild(base, False)
    #create_list_path(base, True)
    #create_list_path(base, False)
    #rename(base, True)
    #rename(base, False)

    modify_gts(base, True)
    modify_gts(base, False)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
