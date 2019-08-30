import torch

from .nets import generator

class FtoA:
    def __init__(self, opts, isTrain=True, isAdv=False):
        self.opts    = opts
        self.isTrain =  isTrain
        self.gen     = generator().cuda()
        self.gen.set_vgg_as_encoder()


        if isTrain:
            self.criteionL1    = torch.nn.L1Loss()
            self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

    def set_inputs(self, inputs, targets):
        self.real_X = torch.cuda.FloatTensor(inputs)
        self.real_Y = torch.cuda.FloatTensor(targets)

    def forward(self):
        self.Z, self.fake_Y = self.gen(self.real_X)
        
    def backward_gen(self):
        #synthetic_pair = torch.cat((self.real_X, self.fake_Y))
        
        self.loss_gen_L1 = self.criteionL1(self.fake_Y, self.real_Y) * self.opts.lambda_L1
        self.loss_gen_L1.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_gen.zero_grad()
        self.backward_gen()
        self.optimizer_gen.step()
        
