import torch
from torch import optim

from models.ftoa_model import FtoA
from options.base import baseOpt
from tools.preprocess import read_and_crop_square_img

def train_op(model, opts):
  ambnt_imgs, flash_imgs = read_and_crop_square_img(path=opts.dataset_path, 
  						                                      mode='train', SIZE=opts.load_size)

    for i in range(0, len(ambnt_imgs), opts.batch_size):
        ambnt_batch = ambnt_imgs[i:i+batch_size]
        flash_batch = flash_imgs[i:i+batch_size]
        
        model.set_inputs(flash_batch, ambnt_batch)        
        model.optimize_parameters()
        
        print(model.Z.size(), model.real_X.size(), model.fake_Y.size())
        break
    
#    print(z)
#    print(z.cpu().detach().numpy())
#    loss = L1()
if __name__ == '__main__':
    opts  = baseOpt().parse()
    model = FtoA(opts)
    
    train_op(model, opts)
