import torch
import numpy as np

from PIL import Image
def PSNR(ifake, iambnt, iflash, it=None, save=True):
    ifake  = ifake.cpu().detach().numpy() 
    iambnt = iambnt.cpu().detach().numpy() 
    iflash = iflash.cpu().detach().numpy() 

    ifake  = np.squeeze(ifake , axis=0)
    iambnt = np.squeeze(iambnt, axis=0)
    iflash = np.squeeze(iflash, axis=0)

    ifake  = np.transpose(ifake , (1, 2, 0))
    iambnt = np.transpose(iambnt, (1, 2, 0))
    iflash = np.transpose(iflash, (1, 2, 0))
    
    ifake  = (ifake *0.5 + 0.5)*255.0
    iambnt = (iambnt*0.5 + 0.5)*255.0
    iflash = (iflash*0.5 + 0.5)*255.0

    ifake  = ifake.astype(np.int32)
    iambnt = iambnt.astype(np.int32)
    iflash = iflash.astype(np.int32)

    if save:
        imfake = Image.fromarray(np.uint8(ifake))
        imreal = Image.fromarray(np.uint8(iambnt))
        im2 = Image.fromarray(np.uint8(iambnt))
        imfake.save('results/fake_it_'+str(it)+'.png')
        imreal.save('results/real_it_'+str(it)+'.png')
    
    MSE_ambnt = np.mean(np.square(ifake - iambnt))
    MSE_flash = np.mean(np.square(ifake - iflash))
    
    PSNR_ambnt = 20.0 * np.log10(255.0) - 10.0 * np.log10(MSE_ambnt)
    PSNR_flash = 20.0 * np.log10(255.0) - 10.0 * np.log10(MSE_flash)

    return PSNR_ambnt, PSNR_flash

