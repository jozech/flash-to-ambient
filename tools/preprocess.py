from __future__ import print_function

import torch
import glob
import numpy as np
import sys

from PIL import Image


def read_pair(a, f):
    img_a = Image.open(a)
    img_f = Image.open(f)
    return img_a, img_f

def dataset_list(path):
    train_ambnt_set = glob.glob(path+'train/*ambient.png')
    train_flash_set = glob.glob(path+'train/*flash.png')

    train_ambnt_set.sort()
    train_flash_set.sort()

    train_set = []

    for i,j in zip(train_ambnt_set,train_flash_set):
        assert (i[:-12] == j[:-10])
        train_set.append([i,j])

    test_ambnt_set = glob.glob(path+'test/*ambient.png')
    test_flash_set = glob.glob(path+'test/*flash.png')

    test_ambnt_set.sort()
    test_flash_set.sort()

    test_set = []
    for i,j in zip(test_ambnt_set,test_flash_set):
        assert (i[:-12] == j[:-10])
        test_set.append([i,j])

    return train_set, test_set

def read_and_crop_square_img(path, mode, SIZE):
    if mode == 'train':
        data_list, _ = dataset_list(path)
    elif mode == 'test':
        _, data_list = dataset_list(path)
    else:
        print("get_data() got an invalid argument for 'mode'('train' or 'test')...")

    flash_list = []
    ambnt_list = []

    n_pairs    = 0
    list_size  = len(data_list)

    print()
    for a, f in data_list:
        img_a, img_f = read_pair(a,f)
        img_a  = img_a.resize([SIZE, SIZE], Image.ANTIALIAS)
        img_f  = img_f.resize([SIZE, SIZE], Image.ANTIALIAS)
        img_a_out    = np.asarray(img_a, dtype=np.float32)/255.0
        img_f_out    = np.asarray(img_f, dtype=np.float32)/255.0

        img_a_out = img_a_out * 2.0 - 1.0
        img_f_out = img_f_out * 2.0 - 1.0

        img_a_out = np.transpose(img_a_out, (2, 0, 1))
        img_f_out = np.transpose(img_f_out, (2, 0, 1))

        ambnt_list.append(img_a_out)
        flash_list.append(img_f_out)

        n_pairs+=1
        print("\rReading: {:3.1f}%".format(100.0*(n_pairs/list_size)), end=' '*4)
    print("\rReading: 100.0%")
    return ambnt_list, flash_list

def L1(X,Y):
    return torch.mean(torch.abs(err_a - err_b))