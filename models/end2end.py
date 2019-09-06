import torch
import os

from .nets import vgg19_generator_deconv, vgg16_generator_unpool, vgg16_generator_deconv

class EDNet:
	def __init__(self, opts, isTrain=True, isAdv=False):
		self.opts    = opts
		self.isTrain =  isTrain
		self.device = torch.device('cuda:{}'.format(self.opts.gpu_ids[0])) if self.opts.gpu_ids else torch.device('cpu') 

		if isTrain:
			print('Training mode![on {}]\n'.format(self.device))
			self.gen    = vgg16_generator_deconv(prob=0.0).cuda()
			self.gen.set_vgg_as_encoder()	
			self.criteionL1    = torch.nn.L1Loss()
			self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
		else:
			print('Testing mode![on {}]\n'.format(self.device))
			self.gen    = vgg16_generator_deconv(prob=0.0).cuda()
			self.gen.set_vgg_as_encoder()
	def set_inputs(self, inputs, targets):
		self.real_X = torch.cuda.FloatTensor(inputs)
		self.real_Y = torch.cuda.FloatTensor(targets)

	def forward(self):
		self.Z, self.fake_Y = self.gen(self.real_X)
		
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
			torch.save(self.gen.cpu().state_dict(), save_path)
			self.gen.cuda()

	def load_model(self, ep):
		file_model = 'model-{}.pth'.format(str(ep))
		load_path  = os.path.join(self.opts.checkpoints_dir, file_model)
		state_dict = torch.load(load_path, map_location=str(self.device))

		self.gen.load_state_dict(state_dict)

class cGAN:
	def __init__(self, opts, isTrain=True, isAdv=False):
		self.opts    = opts
		self.isTrain =  isTrain
		self.device = torch.device('cuda:{}'.format(self.opts.gpu_ids[0])) if self.opts.gpu_ids else torch.device('cpu') 

		if isTrain:
			print('Training mode![on {}]\n'.format(self.device))
			self.Gen = vgg16_generator_unpool(prob=0.0).cuda()
			self.Dis = discriminator().cuda()

			self.Gen.set_vgg_as_encoder()	
			self.criteionL1    = torch.nn.L1Loss()
			self.optimizer_gen = torch.optim.Adam(self.Gen.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
			self.optimizer_dis = torch.optim.Adam(self.Dis.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

		else:
			print('Testing mode![on {}]\n'.format(self.device))
			self.Gen    = vgg16_generator_unpool(prob=0.0).cuda()
			self.Gen.set_vgg_as_encoder()
	def set_inputs(self, inputs, targets):
		self.real_X = torch.cuda.FloatTensor(inputs)
		self.real_Y = torch.cuda.FloatTensor(targets)

	def forward(self):
		self.Z, self.fake_Y = self.Gen(self.real_X)
		
		if self.isTrain:
			synthetic_pair = torch.cat((self.real_X, self.fake_Y))
			authentic_pair = torch.cat((self.real_X, self.real_Y))

			self.dis_out_fake = self.Dis(self.synthetic_pair)
			self.dis_out_real = self.Dis(self.authentic_pair)
			self.CriterionGAN = torch.nn.BCEWithLogitsLoss()
	def backward_gen(self):
		self.loss_gen_L1 = self.criteionL1(self.fake_Y, self.real_Y)
		self.loss_gen    = self.loss_gen_L1 * self.opts.lambda_L1 
		self.loss_gen.backward()

	def backward_dis(self):
		self.CriterionGAN()
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