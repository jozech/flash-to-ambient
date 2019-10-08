from __future__ import print_function

import glob
import numpy as np
import sys
import os
import random
import math

from PIL import Image
from PIL import ImageOps

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

def get_array_list(
    input_list    = None,
    filtered_list = None, 
    crop          = True, 
    load_min_size = None,
    out_size      = None):

    ambnt_list = []
    flash_list = []
    
    ambnt_bf_list = []
    flash_bf_list = []

    for iobj, (img_a, img_f) in enumerate(input_list):
        #img_a.show()
        #img_f.show()
        if filtered_list:
            img_a_bf, img_f_bf = filtered_list[iobj]

        if crop:
            flip_rand = random.random()
            if flip_rand < 0.5:
                img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
                img_f = img_f.transpose(Image.FLIP_LEFT_RIGHT)

            M     = load_min_size * random.uniform(0.8, 1.0)
            wrand = random.randint(0, int(img_a.size[0]-M))
            hrand = random.randint(0, int(img_a.size[1]-M))
 
            img_a = random_crop(img_a, M, wrand, hrand)
            img_f = random_crop(img_f, M, wrand, hrand)
            
            img_a = img_a.resize([out_size, out_size], Image.ANTIALIAS)
            img_f = img_f.resize([out_size, out_size], Image.ANTIALIAS)

            if filtered_list:
                if flip_rand < 0.5:
                    img_a_bf = img_a_bf.transpose(Image.FLIP_LEFT_RIGHT)
                    img_f_bf = img_f_bf.transpose(Image.FLIP_LEFT_RIGHT)    

                img_a_bf = random_crop(img_a_bf, M, wrand, hrand)
                img_f_bf = random_crop(img_f_bf, M, wrand, hrand)
                img_a_bf = img_a_bf.resize([out_size, out_size], Image.ANTIALIAS)
                img_f_bf = img_f_bf.resize([out_size, out_size], Image.ANTIALIAS)


                #img_a_bf.show()
                #img_f_bf.show()

        #img_a.show()
        #img_f.show()

        if filtered_list:
            img_a_bf_out = get_array_to_net(img_a_bf)
            img_f_bf_out = get_array_to_net(img_f_bf)

            ambnt_bf_list.append(img_a_bf_out)
            flash_bf_list.append(img_f_bf_out)

            img_a_bf.close()
            img_f_bf.close()

        img_a_out = get_array_to_net(img_a)
        img_f_out = get_array_to_net(img_f)

        ambnt_list.append(img_a_out)
        flash_list.append(img_f_out)
        
        img_a.close()
        img_f.close()

    data_dict = {
            'ambnt_imgs'    : ambnt_list,
            'flash_imgs'    : flash_list,
            'ambnt_bf_imgs' : ambnt_bf_list,
            'flash_bf_imgs' : flash_bf_list
    }

    return data_dict

def get_array_to_net(im):
    img_arr = np.asarray(im, dtype=np.float32)/255.0
    img_arr = img_arr * 2.0 - 1.0
    img_arr = np.transpose(img_arr, (2, 0, 1))

    return img_arr

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

def shuffle_data(imgs_sets):
    rng_state = np.random.get_state()
    out = []
    print(len(imgs_sets))
    for img_set in imgs_sets:
        np.random.set_state(rng_state)
        np.random.shuffle(img_set)
        out.append(img_set)

    return out


def bilateral_filter(im, win_size, sigma_space=7, sigma_range=102):
    margin  = int(win_size/2)
    im      = ImageOps.expand(im, margin)
    img_arr = np.asarray(im, dtype=np.int32)

    H,W,_ = img_arr.shape
    mask_img = np.zeros((H,W))

    left_bias  = math.floor(-(win_size-1)/2)
    right_bias = math.floor( (win_size-1)/2)
    filtered_img = img_arr.astype(np.float32)

    gaussian_vals = {I: math.exp(-I**2/(2 * sigma_range**2)) for I in range(256)}
    
    gaussian_vals_func   = lambda x: gaussian_vals[x]
    gaussian_vals_matrix = np.vectorize(gaussian_vals_func, otypes=[np.float32])

    space_weights = np.zeros((win_size,win_size,3))

    for i in range(left_bias, right_bias+1):
        for j in range(left_bias, right_bias+1):
            space_weights[i-left_bias][j-left_bias] = math.exp(-(i**2+j**2)/(2*sigma_space**2))
    

    for i in range(margin, H-margin):
        for j in range(margin, W-margin):
            filter_input  = img_arr[i+left_bias:i+right_bias+1, 
                                          j+left_bias:j+right_bias+1]

            range_weights   = gaussian_vals_matrix(np.abs(filter_input-img_arr[i][j]))
            space_and_range = np.multiply(space_weights, range_weights)

            norm_space_and_range = space_and_range / np.sum(space_and_range, keepdims=False, axis=(0,1))
            output = np.sum(np.multiply(norm_space_and_range, filter_input), axis=(0,1))
                
            filtered_img[i][j] = output
    
    filtered_img = np.clip(filtered_img, 0, 255)

    out = filtered_img[margin:-margin, margin:-margin,:].astype(np.uint8)
    return Image.fromarray(out)

def get_filtered_img_objs(im_list):
    img_objs = []

    for iobj, (ambnt_img, flash_img) in enumerate(im_list):
        img_a_bf = bilateral_filter(ambnt_img, 7)
        img_f_bf = bilateral_filter(flash_img, 7)

        img_objs.append([img_a_bf, img_f_bf])

        print("\rfiltering {:3d}/{:3d} ...".format(iobj+1, len(im_list)), end='')
    return img_objs


if __name__ == "__main__":
    im  = Image.open('results/flash_it_3.png')
    out = bilateral_filter(im, 7)

    im.show()
    out.show()
    im.close()
                

                

