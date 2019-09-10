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

from models.end2end import EDNet
from models.end2end import cGAN
from options.base import baseOpt

from tools.pre import read_data, get_array
from tools.post import PSNR

def test_op(model, opts):
    if not os.path.exists('results/'):
        os.makedirs('results/')

    # Make a list of pairs of ambient and flash image filenames
    imgs_list = read_data(path=opts.dataset_path, mode='test')
    # Get array of the images
    ambnt_imgs, flash_imgs = get_array(imgs_list, mode='test')

    psnr_ambnt_it = []
    psnr_flash_it = []
    
    for it in range(1, len(ambnt_imgs)+1):
        # Batch of one
        ambnt_batch = [ambnt_imgs[it-1]]
        flash_batch = [flash_imgs[it-1]]
        
        # Set inputs of the model and run 
        model.set_inputs(flash_batch, ambnt_batch)  
        model.forward()

        # Compute PSNR value of each output image and save on 'results/'
        psnr_ambnt, psnr_flash = PSNR(model.fake_Y, model.real_Y, model.real_X, it, True)
        psnr_ambnt_it.append(psnr_ambnt)
        psnr_flash_it.append(psnr_flash)

        print('\riter:{:4d}/{:4d}, PSNR_it: {:.2f}'.format(it,len(ambnt_imgs), psnr_ambnt_it[-1]), end='')
    print('\rTesting: PSNR: {:.2f}, PSNR(flash): {:.2f}'.format(np.mean(psnr_ambnt_it), np.mean(psnr_flash_it)))

if __name__ == "__main__":
    # Get parameters
    opts  = baseOpt().parse()
    
    # Build model, load, and run test
    model = cGAN(opts, isTrain=False)
    model.load_model(opts.load_epoch)
    test_op(model, opts)