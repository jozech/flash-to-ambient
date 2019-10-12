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
import glob
import numpy as np

from options.base import baseOpt
from tools.post import compute_metrics

def comp_op(opts):
    real_imgs = []

    ground_truth_files = []
    model_output_files = []
    for file in glob.glob('datasets/'+opts.dataset_path+'/test/*ambient.png'):
        subj_file = file[:-11]
        ground_truth_files.append(file)
        model_output_files.append('results/'+opts.model+'/'+subj_file.split('/')[-1]+'flash.png')

    psnr = [] 
    ssim = []
    for a, b in zip(ground_truth_files,model_output_files):
        i_psnr, i_ssim = compute_metrics(a,b)
        psnr.append(i_psnr)
        ssim.append(i_ssim)
    
    print('PSNR: {:.3f}, SSIM: {:.4f}'.format(np.mean(psnr), np.mean(ssim)))
if __name__ == "__main__":
    # Get parameters
    opts  = baseOpt().parse()
    
    print('Computing PSNR and SSIM values on {} model '.format(opts.model))
    comp_op(opts)