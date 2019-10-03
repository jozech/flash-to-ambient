"""
This script works for a encoder-decoder network(EDNet) and for a Conditional Adversarial Network(cGAN). It 
first load the dataset(pairs of filenames):

    img_list = [[object1_ambient.png, object1_flash.png], ...]

Use:

    python test.py --load_epoch=1000
    python test.py --load_epoch=2000
    python test.py --load_epoch=100

See options/base.py for more details about more information of all the default parameters.
"""

import os
import numpy as np

from models.models import setModel

from options.base import baseOpt

from tools.pre import read_data
from tools.pre import get_array_list
from tools.pre import get_array_to_net
from tools.pre import shuffle_data
from tools.pre import get_filtered_img_objs

from tools.post import PSNR

def test_op(model, opts):
    if not os.path.exists('results/'):
        os.makedirs('results/')

    # Make a list of pairs of ambient and flash image filenames
    img_obj_list = read_data(path=opts.dataset_path, mode='test')

    # For the DeepFlash model
    img_bf_obj_list = None
    if opts.model == 'DeepFlash':
        img_bf_obj_list = get_filtered_img_objs(img_obj_list)
    
    # Get array of image objects
    data_dict = get_array_list(input_list=img_obj_list, 
                               filtered_list=img_bf_obj_list, 
                               crop=False)

    ambnt_imgs    = data_dict['ambnt_imgs']
    flash_imgs    = data_dict['flash_imgs']

    # For the DeepFlash model
    if opts.model == 'DeepFlash':
        ambnt_bf_imgs = data_dict['ambnt_bf_imgs']
        flash_bf_imgs = data_dict['flash_bf_imgs']

    psnr_ambnt_it = []
    psnr_flash_it = []
    
    for it in range(0, len(ambnt_imgs)):
        # Batch of one
        ambnt_batch = [ambnt_imgs[it]]
        flash_batch = [flash_imgs[it]]
        
        # Set inputs of the model and run 
        model.set_inputs(flash_batch, ambnt_batch)  
        
        # For the DeepFlash model
        if opts.model == 'DeepFlash':
            ambnt_bf_batch = [ambnt_bf_imgs[it]]
            flash_bf_batch = [flash_bf_imgs[it]]

            # Setting input and target images filtered
            model.set_filtered_inputs(flash_bf_batch, ambnt_bf_batch)  

        model.forward()

        # Compute PSNR value of each output image and save on 'results/'
        psnr_ambnt, psnr_flash = PSNR(model.fake_Y, model.real_Y, model.real_X, it+1, True)
        psnr_ambnt_it.append(psnr_ambnt)
        psnr_flash_it.append(psnr_flash)

        print('\riter:{:4d}/{:4d}, PSNR_it: {:.2f}'.format(it+1,len(ambnt_imgs), psnr_ambnt_it[-1]), end='')
    print('\rTesting: PSNR: {:.2f}, PSNR(flash): {:.2f}'.format(np.mean(psnr_ambnt_it), np.mean(psnr_flash_it)))

if __name__ == "__main__":
    # Get parameters
    opts  = baseOpt().parse()
    
    # Build model, load, and run test
    model, _ = setModel(opts.model, opts)
    model.load_model(opts.load_epoch)
    test_op(model, opts)