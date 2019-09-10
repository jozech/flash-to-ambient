from __future__ import print_function

import torch
import glob
import numpy as np
import sys
import os
import random

from PIL import Image

def read_pair(a, f):
    img_a = Image.open(a)
    img_f = Image.open(f)
    return img_a, img_f

def dataset_list(path):
    source_path  = 'datasets/'
    dataset_path = os.path.join(source_path, path)
    train_ambnt_set = glob.glob(dataset_path + '/train/*ambient.png')
    train_flash_set = glob.glob(dataset_path + '/train/*flash.png')

    train_ambnt_set.sort()
    train_flash_set.sort()

    train_set = []

    for i,j in zip(train_ambnt_set,train_flash_set):
        assert (i[:-12] == j[:-10])
        train_set.append([i,j])

    test_ambnt_set = glob.glob(dataset_path+'/test/*ambient.png')
    test_flash_set = glob.glob(dataset_path+'/test/*flash.png')

    test_ambnt_set.sort()
    test_flash_set.sort()

    test_set = []
    for i,j in zip(test_ambnt_set,test_flash_set):
        assert (i[:-12] == j[:-10])
        test_set.append([i,j])

    return train_set, test_set


def random_crop(img, crop_size, wrand, hrand):
    img = img.crop((wrand, hrand, wrand+crop_size, hrand+crop_size))
    return img

def get_array(
    im_list, 
    mode, 
    MIN_SIZE = None,
    SIZE     = None):

    ambnt_list = []
    flash_list = []

    CROP = True
    if mode == 'test':
        CROP = False

    for img_a, img_f in im_list:
        #img_a.show()
        #img_f.show()
        
        if CROP:
            if random.random() < 0.5:
                img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
                img_f = img_f.transpose(Image.FLIP_LEFT_RIGHT)

            M     = MIN_SIZE * random.uniform(0.8, 1.0)
            wrand = random.randint(0, int(img_a.size[0]-M))
            hrand = random.randint(0, int(img_a.size[1]-M))
 
            img_a = random_crop(img_a, M, wrand, hrand)
            img_f = random_crop(img_f, M, wrand, hrand)
            
            img_a = img_a.resize([SIZE, SIZE], Image.ANTIALIAS)
            img_f = img_f.resize([SIZE, SIZE], Image.ANTIALIAS)
        
        #print(img_a.size)
        #img_a.show()
        #img_f.show()

        img_a_out = np.asarray(img_a, dtype=np.float32)/255.0
        img_f_out = np.asarray(img_f, dtype=np.float32)/255.0

        img_a.close()
        img_f.close()

        img_a_out = img_a_out * 2.0 - 1.0
        img_f_out = img_f_out * 2.0 - 1.0

        img_a_out = np.transpose(img_a_out, (2, 0, 1))
        img_f_out = np.transpose(img_f_out, (2, 0, 1))

        ambnt_list.append(img_a_out)
        flash_list.append(img_f_out)

    return ambnt_list, flash_list

def read_data(path, mode):
    if mode == 'train':
        data_list, _ = dataset_list(path)
    elif mode == 'test':
        _, data_list = dataset_list(path)
    else:
        print("get_data() got an invalid argument for 'mode'('train' or 'test')...")

    im_list = []
    n_pairs    = 0
    list_size  = len(data_list)

    for a, f in data_list:
        img_a_tmp, img_f_tmp = read_pair(a,f)
        img_a = img_a_tmp.copy()
        img_f = img_f_tmp.copy()
        im_list.append([img_a, img_f])
        img_a_tmp.close()
        img_f_tmp.close()
        n_pairs+=1
        print("\rreading data\t: [{:3}/{:3}] {:3.1f}%".format(n_pairs, list_size, 100.0*(n_pairs/list_size)), end='')
    print("\rreading data\t: [{:3}/{:3}] {:3.1f}%".format(n_pairs, list_size, 100.0*(n_pairs/list_size)))
    print("{:s} size\t: {:d} pairs of images".format(mode, len(im_list)), end='\n\n')

    return im_list

def shuffle_data(ambnt_imgs, flash_imgs):
    rng_state = np.random.get_state()
    np.random.shuffle(ambnt_imgs)
    np.random.set_state(rng_state)
    np.random.shuffle(flash_imgs)

    return ambnt_imgs, flash_imgs