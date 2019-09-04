import numpy as np

from models.vgg_ed_model import vgg_encoder_decoder
from options.base import baseOpt

from tools.pre import read_data, get_array
from tools.post import PSNR

def test_op(model, opts):
    imgs_list = read_data(path=opts.dataset_path, mode='test')
    ambnt_imgs, flash_imgs = get_array(imgs_list, mode='test')

    psnr_ambnt_it = []
    psnr_flash_it = []
    
    for it in range(1, len(ambnt_imgs)+1):
        ambnt_batch = [ambnt_imgs[it-1]]
        flash_batch = [flash_imgs[it-1]]
            
        model.set_inputs(flash_batch, ambnt_batch)  
        model.forward()
        psnr_ambnt, psnr_flash = PSNR(model.fake_Y, model.real_Y, model.real_X, it, True)
        psnr_ambnt_it.append(psnr_ambnt)
        psnr_flash_it.append(psnr_flash)

        print('\riter:{:4d}/{:4d}, PSNR_it: {:.2f}'.format(it,len(ambnt_imgs), psnr_ambnt_it[-1]), end='')
    print('\rTesting: PSNR: {:.2f}, PSNR(flash): {:.2f}'.format(np.mean(psnr_ambnt_it), np.mean(psnr_flash_it)))

if __name__ == "__main__":
    opts  = baseOpt().parse()
    model = vgg_encoder_decoder(opts, isTrain=False)
    model.load_model(opts.load_epoch)

    test_op(model, opts)