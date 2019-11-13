import argparse
import os

def str2bool(s):
	if isinstance(s, bool):
		print('is bool')
		return s
	if s.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif s.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

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
		parser.add_argument('--out_act', type=str, default='sigmoid', help='final activation: sigmoid, tanh')
		parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
		parser.add_argument('--lr1', type=float, default=2e-5, help='learning rate for the generator')
		parser.add_argument('--lr2', type=float, default=2e-6, help='learning rate for the discriminator')
		parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyp. for the Adam optimizer')
		parser.add_argument('--lambda_GAN', type=float, default=1.0, help='lambda for the Adversarial Loss')
		parser.add_argument('--R_loss', type=str, default='L1', help='Loss type: Cauchy or L1')
		parser.add_argument('--attention_gen', type=str2bool, default=True, help='Attention mode')
		parser.add_argument('--attention_dis', type=str2bool, default=True, help='Attention mode')
		parser.add_argument('--upsample', type=str, default='deconv', help='upsample mode: deconv, unpool.')
		parser.add_argument('--vgg_freezed', type=str2bool, default=True, help='make or not backpropagation on the the vgg encoder')
		parser.add_argument('--save_epoch', type=int, default=100, help='number of epochs for saving the model')
		parser.add_argument('--load_epoch', type=int, default=0,help='load at epoch #')
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		parser.add_argument('--sample_dir', type=str, default=None, help='sample dir of the sample to eval through the model')
		
		return parser

	def parse(self):	

		parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
		parser = self.initialize(parser)

		opt, _ = parser.parse_known_args()
		return parser.parse_args()