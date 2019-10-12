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