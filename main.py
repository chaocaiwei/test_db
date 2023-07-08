# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import shutil
import argparse


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
            rename_path = gt_path + fid + '.txt'
            shutil.move(cur_path, gt_path)
            os.rename(m_path, rename_path)


def create_list_path(base, training):

    source_path = base + 'train_images/' if training else base + 'test_images/'
    list_path = base + 'train_list.txt' if training else base + 'test_list.txt'

    files = os.listdir(source_path)
    with open(list_path, 'w') as fd:
        for file in files:
            fd.write(file + '\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('base', type=str)
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    base = args['base']

    mov_fild(base, True)
    mov_fild(base, False)
    create_list_path(base, True)
    create_list_path(base, False)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
