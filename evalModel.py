import os
import numpy as np
import time

from models.models import setModel

from options.base import baseOpt
from tools.post import saveimg
from tools.pre import get_array_to_net
from PIL import Image

def eval_op(model, opts):
    results_path = 'results/single/'+opts.model+'_'+opts.upsample+'_'+opts.out_act+'_attgen_'+str(opts.attention_gen)+'_attdis_'+str(opts.attention_dis)+'_epoch-'+str(opts.load_epoch)+'/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    t_start = time.time()
    sample_obj_img = Image.open(opts.sample_dir)
    sample_img     = get_array_to_net(sample_obj_img, opts.out_act)
    # Set inputs of the model and run 
    model.set_inputs([sample_img], None)  
    model.forward()
    saveimg(results_path, opts.sample_dir.split('/')[-1], model.fake_Y, opts.out_act)
    print('New sample convereted...')

if __name__ == "__main__":
    # Get parameters
    opts  = baseOpt().parse()
    
    # Build model, load, and run test
    print('Running {} model on one sample'.format(opts.model))
    model, _ = setModel(opts, False)

    model.load_model(opts.load_epoch)
    eval_op(model, opts)