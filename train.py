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

from models.end2end import EDNet
from models.end2end import cGAN

from options.base import baseOpt

from tools.pre import read_data, get_array
from tools.pre import shuffle_data

import numpy as np
import time

def train_op(model, opts):
    # Make a list of pairs of ambient and flash image filenames
    imgs_list  = read_data(path=opts.dataset_path, mode='train')
    train_size = len(imgs_list)

    for ep in range(1, opts.epochs+1):
        start = time.time()
        # Get array of the images, make data augmentation and random shuffle
        ambnt_imgs, flash_imgs = get_array(imgs_list, mode='train', MIN_SIZE=opts.load_size, SIZE=opts.crop_size)
        ambnt_imgs, flash_imgs = shuffle_data(ambnt_imgs, flash_imgs)

        loss_it  = []
        loss_gen = []
        loss_dis = []

        for it in range(0, len(ambnt_imgs), opts.batch_size):
            # Batch of images
            ambnt_batch = ambnt_imgs[it:it+opts.batch_size]
            flash_batch = flash_imgs[it:it+opts.batch_size]
            
            # Set inputs of the model and run 
            model.set_inputs(flash_batch, ambnt_batch)       
            model.optimize_parameters()

            loss_it.append(model.loss_gen_L1.cpu().detach().numpy())
            loss_gen.append(model.loss_gen_GAN.cpu().detach().numpy())
            loss_dis.append(model.loss_Dis.cpu().detach().numpy())
            print('\riter:{:4d}/{:4d}, loss_batch(L1): {:.4f}, loss_gen: {:.4f}, loss_dis: {:.4f}'.format(int(it+opts.batch_size),train_size,loss_it[-1], loss_gen[-1], loss_dis[-1]), end='')
        end = time.time()
        print('\repochs: {:4d}, loss_batch(L1):{:.4f}, loss_gen: {:.4f}, loss_dis: {:.4f} in {:3.2f}s'.format(ep, np.mean(loss_it), np.mean(loss_gen), np.mean(loss_dis),(end-start)))

        # Save model each {opts.save_epoch} epochs
        if ep % opts.save_epoch == 0: 
            print('saving model at epoch {:4d}'.format(ep))
            model.save_model(ep)

if __name__ == '__main__':
    # Get parameters
    opts  = baseOpt().parse()
    
    # Build model, and run test
    model = cGAN(opts)

    # Uncomment for loading a model
    #model.load_model(opts.load_epoch)
    
    train_op(model, opts)