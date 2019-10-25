"""
This script works for a encoder-decoder network(EDNet) and for a Conditional Adversarial Network(cGAN). It 
first load the dataset(pairs of filenames).

Use:

    python test.py --load_epoch=1000
    python test.py --load_epoch=2000
    python test.py --load_epoch=100

See options/base.py for more details about more information of all the default parameters.
"""

import os
import numpy as np
import time

from models.models import setModel

from options.base import baseOpt

from tools.pre import read_test_data
from tools.pre import get_array_list_on_test
from tools.pre import shuffle_data
from tools.pre import get_filtered_img_objs
from tools.post import saveimg

def test_op(model, opts):
    results_path = 'results/'+opts.model+'_'+opts.upsample+'_'+opts.out_act+'_attgen_'+str(opts.attention_gen)+'_attdis_'+str(opts.attention_dis)+'_epoch-'+str(opts.load_epoch)+'/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    t_start = time.time()

    # Make a list of pairs of ambient and flash image filenames
    file_list, img_obj_list = read_test_data(path=opts.dataset_path)

    # For the DeepFlash model
    img_bf_obj_list = None
    if opts.model == 'DeepFlash':
        img_bf_obj_list = get_filtered_img_objs(img_obj_list)

    # Get array of image objects
    data_dict = get_array_list_on_test(input_list    = img_obj_list, 
                                       filtered_list = img_bf_obj_list,
                                       out_act       = opts.out_act)

    t_end  = time.time()
    t_prep = (t_end - t_start)/(2 * len(img_obj_list))

    flash_imgs    = data_dict['flash_imgs']

    # For the DeepFlash model
    if opts.model == 'DeepFlash':
        flash_bf_imgs = data_dict['flash_bf_imgs']

    
    for it in range(0, len(flash_imgs)):
        # Batch of one
        #ambnt_batch = [ambnt_imgs[it]]
        flash_batch = [flash_imgs[it]]
        flash_file  = file_list[it]

        # Set inputs of the model and run 
        model.set_inputs(flash_batch, None)  
        
        # For the DeepFlash model
        if opts.model == 'DeepFlash':
            flash_bf_batch = [flash_bf_imgs[it]]

            # Setting input and target images filtered
            model.set_filtered_inputs(flash_bf_batch, None)  

        model.forward()
        saveimg(results_path, flash_file, model.fake_Y, opts.out_act)
        print('\riter:{:4d}/{:4d}'.format(it+1,len(flash_imgs)), end='')
    print('\rTesting [{:4d}/{:4d}]: check the results on "{}"'.format(it+1,len(flash_imgs), results_path))

if __name__ == "__main__":
    # Get parameters
    opts  = baseOpt().parse()
    
    # Build model, load, and run test
    print('Testing {} model '.format(opts.model))
    model, _ = setModel(opts, False)

    model.load_model(opts.load_epoch)
    test_op(model, opts)