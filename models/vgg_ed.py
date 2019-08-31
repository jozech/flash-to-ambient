import torch
import os

from .nets import generator_concat, generator_unpool

class vgg_encoder_decoder:
	def __init__(self, opts, isTrain=True, isAdv=False):
		self.opts    = opts
		self.isTrain =  isTrain


		self.device = torch.device('cuda:{}'.format(self.opts.gpu_ids[0])) if self.opts.gpu_ids else torch.device('cpu') 
		self.gen    = generator_concat().cuda()
		self.gen.set_vgg_as_encoder()

		if isTrain:
			print('Training mode![on {}]\n'.format(self.device))
			self.criteionL1    = torch.nn.L1Loss()
			self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

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