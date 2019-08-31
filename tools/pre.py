from __future__ import print_function

import torch
import glob
import numpy as np
import sys
import os
from PIL import Image


def read_pair(a, f):
    img_a = Image.open(a)
    img_f = Image.open(f)
    return img_a, img_f

def dataset_list(path):
    source_path  = 'datasets/'
    dataset_path = os.path.join(source_path, path)
    train_ambnt_set = glob.glob(dataset_path + 'train/*ambient.png')
    train_flash_set = glob.glob(dataset_path + 'train/*flash.png')

    train_ambnt_set.sort()
    train_flash_set.sort()

    train_set = []

    for i,j in zip(train_ambnt_set,train_flash_set):
        assert (i[:-12] == j[:-10])
        train_set.append([i,j])

    test_ambnt_set = glob.glob(dataset_path+'test/*ambient.png')
    test_flash_set = glob.glob(dataset_path+'test/*flash.png')

    test_ambnt_set.sort()
    test_flash_set.sort()

    test_set = []
    for i,j in zip(test_ambnt_set,test_flash_set):
        assert (i[:-12] == j[:-10])
        test_set.append([i,j])

    return train_set, test_set

def read_data(path, mode, SIZE):
    if mode == 'train':
        data_list, _ = dataset_list(path)
        RESIZE = True
    elif mode == 'test':
        _, data_list = dataset_list(path)
        RESIZE = False
    else:
        print("get_data() got an invalid argument for 'mode'('train' or 'test')...")

    flash_list = []
    ambnt_list = []

    n_pairs    = 0
    list_size  = len(data_list)

    for a, f in data_list:
        img_a, img_f = read_pair(a,f)
        
        if RESIZE:
            img_a  = img_a.resize([SIZE, SIZE], Image.ANTIALIAS)
            img_f  = img_f.resize([SIZE, SIZE], Image.ANTIALIAS)
        img_a_out = np.asarray(img_a, dtype=np.float32)/255.0
        img_f_out = np.asarray(img_f, dtype=np.float32)/255.0

        img_a_out = img_a_out * 2.0 - 1.0
        img_f_out = img_f_out * 2.0 - 1.0

        img_a_out = np.transpose(img_a_out, (2, 0, 1))
        img_f_out = np.transpose(img_f_out, (2, 0, 1))

        ambnt_list.append(img_a_out)
        flash_list.append(img_f_out)

        n_pairs+=1
        print("\rreading data {:3.1f}%".format(100.0*(n_pairs/list_size)), end=' '*4)
    print("\rreading data 100.0% ")
    return ambnt_list, flash_list

def shuffle_data(ambnt_imgs, flash_imgs):
    rng_state = np.random.get_state()
    np.random.shuffle(ambnt_imgs)
    np.random.set_state(rng_state)
    np.random.shuffle(flash_imgs)

    return ambnt_imgs, flash_imgs