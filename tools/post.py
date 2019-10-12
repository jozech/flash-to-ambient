import torch
import numpy as np

from PIL import Image
from skimage.measure import compare_ssim

def compute_metrics(ambient_file, model_result):
    tar_img = Image.open(ambient_file)
    out_img = Image.open(model_result)

    tar_img_ar = np.array(tar_img, dtype=np.int32)
    out_img_ar = np.array(out_img, dtype=np.int32)

    MSE  = np.mean(np.square(tar_img_ar - out_img_ar))
    PSNR = 20.0 * np.log10(255.0) - 10.0 * np.log10(MSE)

    tar_img_gray = tar_img.convert('L')
    out_img_gray = out_img.convert('L')

    tar_img_gray_ar = np.array(tar_img_gray, dtype=np.uint8)
    out_img_gray_ar = np.array(out_img_gray, dtype=np.uint8)

    SSIM = compare_ssim(tar_img_gray_ar, out_img_gray_ar)

    return PSNR, SSIM

def saveimg(results_path, full_file, ifake):
    ifake  = ifake.cpu().detach().numpy() 
    ifake  = np.squeeze(ifake , axis=0)
    ifake  = np.transpose(ifake , (1, 2, 0))
    ifake  = (ifake *0.5 + 0.5)*255.0
    ifake  = ifake.astype(np.uint8)

    f = full_file.split('/')[-1]
    imfake = Image.fromarray(ifake)
    imfake.save(results_path + f)
    imfake.close()