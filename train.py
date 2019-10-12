"""
This script works for a encoder-decoder network(EDNet) and a Conditional Adversarial Network(cGAN). It 
first load the dataset(pairs of filenames):

    img_list = [[object1_ambient.png, object1_flash.png], ...]

Next, we read all the images, perform data augmentation and suffle the list of images on each epoch. Finally,
we run the model on each batch of images.

Use:

    python train.py
    python train.py --batch-size=32
    python train.py --batch-size=64 --save_epoch=1000

See options/base.py for more details about more information of all the default parameters.
"""

from models.models import setModel
from options.base import baseOpt

from tools.pre import read_train_data
from tools.pre import get_array_list
from tools.pre import get_array_to_net
from tools.pre import shuffle_data
from tools.pre import get_filtered_img_objs

import numpy as np
import time
import os
from PIL import Image

def train_op(model, opts, isAdv):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Make a list of pairs of ambient and flash image filenames
    img_obj_list = read_train_data(path=opts.dataset_path)
    train_size   = len(img_obj_list)
    indices      = np.arange(train_size)
    
    img_bf_obj_list = None
    if opts.model == 'DeepFlash':
        img_bf_obj_list = get_filtered_img_objs(img_obj_list)

    if opts.vgg_freezed == False and opts.model != 'UNet':
        print('Unfreezing vgg_encoder...\n')
        model.set_requires_grad(model.Gen, requires_grad=True)

    for ep in range(opts.load_epoch+1, opts.load_epoch+opts.epochs+1):
        start = time.time()
        # Get array of the images, make data augmentation and random shuffle
        data_dict = get_array_list(input_list    = img_obj_list, 
                                   filtered_list = img_bf_obj_list,
                                   load_min_size = opts.load_size, 
                                   out_size      = opts.crop_size)

        ambnt_imgs = np.array(data_dict['ambnt_imgs'])
        flash_imgs = np.array(data_dict['flash_imgs'])

        
        np.random.shuffle(indices)
        
        ambnt_imgs = ambnt_imgs[indices]
        flash_imgs = flash_imgs[indices]

        # For the DeepFlash model
        if opts.model == 'DeepFlash':
            ambnt_bf_imgs = np.array(data_dict['ambnt_bf_imgs'])
            flash_bf_imgs = np.array(data_dict['flash_bf_imgs'])
            ambnt_bf_imgs = ambnt_bf_imgs[indices]
            flash_bf_imgs = flash_bf_imgs[indices]          
        
        loss_it  = []
        loss_gen = []
        loss_dis = []

        for it in range(0, len(ambnt_imgs), opts.batch_size):
            # Batch of images
            ambnt_batch = ambnt_imgs[it:it+opts.batch_size]
            flash_batch = flash_imgs[it:it+opts.batch_size]
            
            # Set inputs of the model and run 
            model.set_inputs(flash_batch, ambnt_batch)       
            
            # For the DeepFlash model
            if opts.model == 'DeepFlash':
                ambnt_bf_batch = ambnt_bf_imgs[it:it+opts.batch_size]
                flash_bf_batch = flash_bf_imgs[it:it+opts.batch_size]    
                model.set_filtered_inputs(flash_bf_batch, ambnt_bf_batch)              

            model.optimize_parameters()
            loss_it.append(model.loss_R.cpu().detach().numpy())

            # Reporting loss value
            print('\riter:{:4d}/{:4d}, loss_batch(R): {:.4f}'.format(int(it+opts.batch_size),train_size,loss_it[-1]), end='')
            
            if isAdv:
                loss_gen.append(model.loss_Gen.cpu().detach().numpy())
                loss_dis.append(model.loss_Dis.cpu().detach().numpy()/2)
                print(', loss_gen: {:.4f}, loss_dis: {:.4f}'.format(loss_gen[-1], loss_dis[-1]), end='')
           
        end = time.time()

        print('\repochs: {:4d}, loss_batch(R):{:.4f}'.format(ep, np.mean(loss_it)), end='')
        if isAdv:
            print(', loss_gen: {:.4f}, loss_dis: {:.4f}'.format(np.mean(loss_gen), np.mean(loss_dis)), end='')
        print(' in {:3.2f}s'.format(end-start))

        # Save model each {opts.save_epoch} epochs
        if ep % opts.save_epoch == 0: 
            print('saving model at epoch {:4d}'.format(ep))
            model.save_model(ep)

if __name__ == '__main__':
    # Get parameters
    opts  = baseOpt().parse()
    
    # Build model, and run test
    model, isAdv = setModel(opts)

    # Loading a model
    if opts.load_epoch > 0:
        print('Loading model at epoch {:d}'.format(opts.load_epoch))
        model.load_model(opts.load_epoch)
    train_op(model, opts, isAdv)