from models.vgg_ed_model import vgg_encoder_decoder
from options.base import baseOpt

from tools.pre import read_data, get_array
from tools.pre import shuffle_data

import numpy as np

def train_op(model, opts):
    imgs_list = read_data(path=opts.dataset_path, mode='train')

    for ep in range(1, opts.epochs+1):
        ambnt_imgs, flash_imgs = get_array(imgs_list, mode='train', SIZE=opts.crop_size)
        ambnt_imgs, flash_imgs = shuffle_data(ambnt_imgs, flash_imgs)
        #return
        nimgs = 0 
        loss_it = []
        for it in range(0, len(ambnt_imgs), opts.batch_size):
            ambnt_batch = ambnt_imgs[it:it+opts.batch_size]
            flash_batch = flash_imgs[it:it+opts.batch_size]
            
            model.set_inputs(flash_batch, ambnt_batch)       
            model.optimize_parameters()

            nimgs+=len(ambnt_batch)
            loss_it.append(model.loss_gen_L1.cpu().detach().numpy())
            print('\riter:{:4d}/{:4d}, loss_batch: {:.4f}'.format(nimgs,len(ambnt_imgs),loss_it[-1]), end='')
        print('\repochs: {:4d}, loss_epoch:{:.5f}'.format(ep, np.mean(loss_it)))

        if ep % opts.save_epoch == 0: 
            print('saving model at epoch {:4d}'.format(ep))
            model.save_model(ep)

if __name__ == '__main__':
    opts  = baseOpt().parse()
    model = vgg_encoder_decoder(opts)
    
    train_op(model, opts)