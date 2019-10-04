import argparse
import os

class baseOpt:
	def __init__(self):
		self.init = False 
	
	def initialize(self, parser):
		parser.add_argument('--dataset_path', default='DATASET_LR', help='path to pairs of images with subfulders train and test')
		parser.add_argument('--model', default='advModel', help='model: advModel, advModelMOD, DeepFlash, ...')
		parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
		parser.add_argument('--load_size', type=int, default=240, help='crop step')
		parser.add_argument('--crop_size', type=int, default=224, help='crop step')
		parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
		parser.add_argument('--lr1', type=float, default=2e-4, help='number of epochs')
		parser.add_argument('--lr2', type=float, default=2e-5, help='number of epochs')
		parser.add_argument('--beta1', type=float, default=0.5, help='number of epochs')
		parser.add_argument('--lambda_GAN', type=float, default=1.0, help='number of epochs')
		parser.add_argument('--vgg_freezed', type=bool, default=True, help='make or not backpropagation on the the vgg encoder')
		parser.add_argument('--save_epoch', type=int, default=0.01, help='number of epochs for saving the model')
		parser.add_argument('--load_epoch', type=int, default=0,help='load at epoch #')
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		
		return parser

	def parse(self):	

		parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
		parser = self.initialize(parser)

		opt, _ = parser.parse_known_args()
		return parser.parse_args()