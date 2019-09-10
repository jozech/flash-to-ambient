import torch
import os
import numpy as np

from torchvision import transforms

from .nets import vgg16_generator_unpool
from .nets import vgg16_generator_deconv
from .nets import discriminator
from .nets import GANLoss

class EDNet:
	def __init__(self, opts, isTrain=True, isAdv=False):
		self.opts    = opts
		self.isTrain =  isTrain
		self.device = torch.device('cuda:{}'.format(self.opts.gpu_ids[0])) if self.opts.gpu_ids else torch.device('cpu') 

		if isTrain:
			print('Training mode![on {}]\n'.format(self.device))
			self.Gen = vgg16_generator_deconv(levels=5).cuda()
			self.Gen.set_vgg_as_encoder()	
			
			self.criteionL1    = torch.nn.L1Loss()
			self.optimizer_gen = torch.optim.Adam(self.Gen.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
		else:
			print('Testing mode![on {}]\n'.format(self.device))
			self.Gen    = vgg16_generator_deconv(levels=5).cuda()
			self.Gen.set_vgg_as_encoder()

	def set_inputs(self, inputs, targets, augmentation=True):
		self.real_X = torch.cuda.FloatTensor(inputs)
		self.real_Y = torch.cuda.FloatTensor(targets)

	def forward(self):
		self.Z, self.fake_Y = self.Gen(self.real_X)
		
	def backward_gen(self):
		#synthetic_pair = torch.cat((self.real_X, self.fake_Y))
		
		self.loss_gen_L1 = self.criteionL1(self.fake_Y, self.real_Y)
		self.loss_gen    = self.loss_gen_L1 * self.opts.lambda_L1 
		self.loss_gen.backward()

	def optimize_parameters(self):
		self.optimizer_gen.zero_grad()
		self.forward()
		self.backward_gen()
		self.optimizer_gen.step()

	def save_model(self, ep):
		file_model = 'model-{}.pth'.format(str(ep))
		save_path = os.path.join(self.opts.checkpoints_dir, file_model)

		if len(self.opts.gpu_ids) > 0 and torch.cuda.is_available():
			torch.save(self.Gen.cpu().state_dict(), save_path)
			self.Gen.cuda()

	def load_model(self, ep):
		file_model = 'model-{}.pth'.format(str(ep))
		load_path  = os.path.join(self.opts.checkpoints_dir, file_model)
		state_dict = torch.load(load_path, map_location=str(self.device))

		self.Gen.load_state_dict(state_dict)

class cGAN:
	def __init__(self, opts, isTrain=True, isAdv=False):
		self.opts    = opts
		self.isTrain =  isTrain
		self.device = torch.device('cuda:{}'.format(self.opts.gpu_ids[0])) if self.opts.gpu_ids else torch.device('cpu') 

		if isTrain:
			print('Training mode![on {}]\n'.format(self.device))
			self.Gen = vgg16_generator_deconv(levels=5).cuda()
			self.Dis = discriminator().cuda()
			self.Gen.set_vgg_as_encoder()	

			self.criterionGAN  = GANLoss().cuda()
			self.criteionL1    = torch.nn.L1Loss()
			self.optimizer_gen = torch.optim.Adam(self.Gen.parameters(), lr=opts.lr1, betas=(opts.beta1, 0.999))
			self.optimizer_dis = torch.optim.Adam(self.Dis.parameters(), lr=opts.lr2, betas=(opts.beta1, 0.999))
		else:
			print('Testing mode![on {}]\n'.format(self.device))
			self.Gen    = vgg16_generator_deconv(levels=5).cuda()
			self.Gen.set_vgg_as_encoder()

	def set_inputs(self, inputs, targets, augmentation=True):
		self.real_X = torch.cuda.FloatTensor(inputs)
		self.real_Y = torch.cuda.FloatTensor(targets)

	def forward(self):
		self.Z, self.fake_Y = self.Gen(self.real_X)

	def backward_gen(self):
		synthetic_pair = torch.cat((self.real_X, self.fake_Y))
		dis_out_fake   = self.Dis(synthetic_pair)

		# We set mode=real, because we will use the first term of the BCEWithLogitsLoss
		self.loss_gen_GAN = self.criterionGAN(dis_out_fake, 'real')
		self.loss_gen_L1  = self.criteionL1(self.fake_Y, self.real_Y)

		self.loss_Gen = self.loss_gen_L1 * self.opts.lambda_L1 + self.loss_gen_GAN
		self.loss_Gen.backward()	

	def backward_dis(self):
		synthetic_pair = torch.cat((self.real_X, self.fake_Y))
		authentic_pair = torch.cat((self.real_X, self.real_Y))

		# No backpropagation along the generator (detach)
		dis_out_fake = self.Dis(synthetic_pair.detach())
		dis_out_real = self.Dis(authentic_pair)
		
		
		self.loss_dis_fake = self.criterionGAN(dis_out_fake, 'fake')
		self.loss_dis_real = self.criterionGAN(dis_out_real, 'real')

		self.loss_Dis = (self.loss_dis_fake + self.loss_dis_real) * 0.5
		self.loss_Dis.backward()

	def optimize_parameters(self):
		# Update Discriminator
		self.forward()
		self.set_requires_grad(self.Dis, True)
		self.optimizer_dis.zero_grad()
		self.backward_dis()
		self.optimizer_dis.step()

		# Update Generator
		self.set_requires_grad(self.Dis, False)
		self.optimizer_gen.zero_grad()
		self.backward_gen()
		self.optimizer_gen.step()

	def set_requires_grad(self, nets, requires_grad=False):
		"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def save_model(self, ep):
		file_model = 'model-{}.pth'.format(str(ep))
		save_path = os.path.join(self.opts.checkpoints_dir, file_model)

		if len(self.opts.gpu_ids) > 0 and torch.cuda.is_available():
			torch.save(self.Gen.cpu().state_dict(), save_path)
			self.Gen.cuda()

	def load_model(self, ep):
		file_model = 'model-{}.pth'.format(str(ep))
		load_path  = os.path.join(self.opts.checkpoints_dir, file_model)
		state_dict = torch.load(load_path, map_location=str(self.device))

		self.Gen.load_state_dict(state_dict)