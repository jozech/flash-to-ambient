import torch
import torch.nn as nn

from .vgg import vgg16_encoder, vgg16_decoder

class vgg16_generator_unpool(nn.Module):
    def __init__(self, prob):
        super(vgg16_generator_unpool, self).__init__()

        self.levels = 5
        self.enc5   = vgg16_encoder(5)
        self.dec5   = vgg16_decoder(5, prob)

    def forward(self, input_imgs):  
        
        layers = self.enc5(input_imgs)
        out    = self.dec5(layers)
            
        return layers['z'], out

    def set_vgg_as_encoder(self):
        from torchvision import models
        
        vgg16 = models.vgg16(pretrained=True, progress=True)
        features_list = list(vgg16.features)

        with torch.no_grad():
            # [224x224]
            if self.levels > 1:
                self.enc5.conv1_1.weight.copy_(features_list[0].weight)
                self.enc5.conv1_1.bias.copy_(features_list[0].bias)
                self.enc5.conv1_2.weight.copy_(features_list[2].weight)
                self.enc5.conv1_2.bias.copy_(features_list[2].bias)
            
            # [112x112]
            if self.levels > 2:
                self.enc5.conv2_1.weight.copy_(features_list[5].weight)
                self.enc5.conv2_1.bias.copy_(features_list[5].bias)
                self.enc5.conv2_2.weight.copy_(features_list[7].weight)
                self.enc5.conv2_2.bias.copy_(features_list[7].bias)

            # [56x56]
            if self.levels > 3:
                self.enc5.conv3_1.weight.copy_(features_list[10].weight)
                self.enc5.conv3_1.bias.copy_(features_list[10].bias)
                self.enc5.conv3_2.weight.copy_(features_list[12].weight)
                self.enc5.conv3_2.bias.copy_(features_list[12].bias)
                self.enc5.conv3_3.weight.copy_(features_list[14].weight)
                self.enc5.conv3_3.bias.copy_(features_list[14].bias)

            # [28x28]
            if self.levels > 4:
                self.enc5.conv4_1.weight.copy_(features_list[17].weight)
                self.enc5.conv4_1.bias.copy_(features_list[17].bias)
                self.enc5.conv4_2.weight.copy_(features_list[19].weight)
                self.enc5.conv4_2.bias.copy_(features_list[19].bias)
                self.enc5.conv4_3.weight.copy_(features_list[21].weight)
                self.enc5.conv4_3.bias.copy_(features_list[21].bias)

            # [14x14]
            if self.levels > 5:
                self.enc5.conv5_1.weight.copy_(features_list[24].weight)
                self.enc5.conv5_1.bias.copy_(features_list[24].bias)
                self.enc5.conv5_2.weight.copy_(features_list[26].weight)
                self.enc5.conv5_2.bias.copy_(features_list[26].bias)
                self.enc5.conv5_3.weight.copy_(features_list[28].weight)
                self.enc5.conv5_3.bias.copy_(features_list[28].bias)

class vgg16_generator_deconv(nn.Module):
    def __init__(self, levels):
        super(vgg16_generator_deconv, self).__init__()
        assert (levels > 0)

        self.enc5   = vgg16_encoder(levels=levels)
        self.dec5   = vgg16_decoder(levels=levels)
        self.levels = levels

    def forward(self, input_imgs):  

        layers  = self.enc5.forward_multiple(input_imgs)
        gray_img= input_imgs.mean(dim=1, keepdim=True)
        out_img = self.dec5.forward_deconv(layers,gray_img=gray_img)
            
        return layers['z'], out_img

    def set_vgg_as_encoder(self):
        from torchvision import models
        
        vgg16 = models.vgg16(pretrained=True, progress=True)
        features_list = list(vgg16.features)

        with torch.no_grad():

            if self.levels > 0:
                # [224x224]
                self.enc5.conv1_1.weight.copy_(features_list[0].weight)
                self.enc5.conv1_1.bias.copy_(features_list[0].bias)
                self.enc5.conv1_2.weight.copy_(features_list[2].weight)
                self.enc5.conv1_2.bias.copy_(features_list[2].bias)
            
            if self.levels > 1:
                # [112x112]
                self.enc5.conv2_1.weight.copy_(features_list[5].weight)
                self.enc5.conv2_1.bias.copy_(features_list[5].bias)
                self.enc5.conv2_2.weight.copy_(features_list[7].weight)
                self.enc5.conv2_2.bias.copy_(features_list[7].bias)

            if self.levels > 2:
                # [56x56]
                self.enc5.conv3_1.weight.copy_(features_list[10].weight)
                self.enc5.conv3_1.bias.copy_(features_list[10].bias)
                self.enc5.conv3_2.weight.copy_(features_list[12].weight)
                self.enc5.conv3_2.bias.copy_(features_list[12].bias)
                self.enc5.conv3_3.weight.copy_(features_list[14].weight)
                self.enc5.conv3_3.bias.copy_(features_list[14].bias)

            if self.levels > 3:
                # [28x28]
                self.enc5.conv4_1.weight.copy_(features_list[17].weight)
                self.enc5.conv4_1.bias.copy_(features_list[17].bias)
                self.enc5.conv4_2.weight.copy_(features_list[19].weight)
                self.enc5.conv4_2.bias.copy_(features_list[19].bias)
                self.enc5.conv4_3.weight.copy_(features_list[21].weight)
                self.enc5.conv4_3.bias.copy_(features_list[21].bias)

            if self.levels > 4:
                # [14x14]
                self.enc5.conv5_1.weight.copy_(features_list[24].weight)
                self.enc5.conv5_1.bias.copy_(features_list[24].bias)
                self.enc5.conv5_2.weight.copy_(features_list[26].weight)
                self.enc5.conv5_2.bias.copy_(features_list[26].bias)
                self.enc5.conv5_3.weight.copy_(features_list[28].weight)
                self.enc5.conv5_3.bias.copy_(features_list[28].bias)

class discriminator(nn.Module):
    def __init__(
        self, 
        init_ch    = 64, 
        ksize      = 3, 
        down_leves = 3, 
        deep       = 5,
        att        = False):
        
        """
            Discriminator:

            stride  : 2-2-2-1
            channels: 64-128-256-512
        """
        super(discriminator, self).__init__()

        seq   = []
        in_ch  = 6
        out_ch = init_ch
        pad    = int((ksize/2 - 1) if ksize%2 == 0 else (ksize - 1)/2)

        self.att = att

        for _ in range(down_leves):
            seq += self.conv_block(in_ch, out_ch, ksize, 2, pad)
            in_ch  = out_ch
            out_ch = out_ch*2

        for _ in range(deep-down_leves-1):
            seq += self.conv_block(in_ch, out_ch, ksize, 1, pad)
            in_ch  = out_ch
            out_ch = out_ch*2

        seq.append(nn.Conv2d(in_channels = in_ch,
                             out_channels= 1,
                             kernel_size = ksize,
                             stride      = 1,
                             padding     = pad))

        self.dis_arch = nn.Sequential(*seq)
        

    def conv_block(self, in_ch, out_ch, ksize, stride, pad):
        subseq = [nn.Conv2d(in_channels  = in_ch,
                            out_channels = out_ch,
                            kernel_size  = ksize, 
                            stride       = stride, 
                            padding      = pad)]

        if in_ch != 6:
            subseq.append(nn.BatchNorm2d(out_ch))
        subseq.append(nn.LeakyReLU(inplace=True))

        return  subseq


    def forward(self, input_pair, att_map=None):
        if self.att:
            input_pair = torch.mul(input_pair, att_map)
        out = self.dis_arch(input_pair) #[28x28]
        return out

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, pred, mode):
        if mode == 'real':
            target = self.real_label
        elif mode == 'fake':
            target = self.fake_label

        target = target.expand_as(pred)
        out    = self.loss(pred, target)
        
        return out


class UNet(nn.Module):
    """
        U-Net architecture based on "Learning to See in the Dark" (Chen et al., 2018)

        arXiv: https://arxiv.org/abs/1805.01934
    """

    def __init__(self, img_ch=3, init_ch=32, levels=5):
        super(UNet, self).__init__()
        self.init_ch = init_ch
        self.levels  = levels

        if levels > 0:
            self.conv_block1 = self.conv_block(img_ch, init_ch)
        
        if levels > 1:
            self.maxpool1    = nn.MaxPool2d(2,2)
            self.conv_block2 = self.conv_block(init_ch, init_ch*2)
            
        if levels > 2:
            self.maxpool2    = nn.MaxPool2d(2,2)
            self.conv_block3 = self.conv_block(init_ch*2, init_ch*4)
            
        if levels > 3:
            self.maxpool3    = nn.MaxPool2d(2,2)
            self.conv_block4 = self.conv_block(init_ch*4, init_ch*8)
            
        if levels > 4:
            self.maxpool4    = nn.MaxPool2d(2,2)
            self.conv_block5 = self.conv_block(init_ch*8, init_ch*16)
            self.deconv4       = nn.ConvTranspose2d(init_ch*16, init_ch*8, 2, stride=2)
            
        if levels > 3:
            self.upconv_block4 = self.conv_block(init_ch*16, init_ch*8)
            self.deconv3       = nn.ConvTranspose2d(init_ch*8, init_ch*4, 2, stride=2)

        if levels > 2:    
            self.upconv_block3 = self.conv_block(init_ch*8, init_ch*4)
            self.deconv2       = nn.ConvTranspose2d(init_ch*4, init_ch*2, 2, stride=2)
        
        if levels > 1:    
            self.upconv_block2 = self.conv_block(init_ch*4, init_ch*2)              
            self.deconv1       = nn.ConvTranspose2d(init_ch*2, init_ch, 2, stride=2)

        if levels > 0:
            self.upconv_block1 = self.conv_block(init_ch*2, init_ch)      
            self.out_conv      = nn.Conv2d(init_ch, img_ch, 1, padding=0)
            self.out_act       = nn.Tanh()

    def conv_block(self, in_ch, out_ch):
        return  nn.Sequential(
                    nn.Conv2d(in_ch , out_ch, 3, padding=1),
                    #nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    #nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )

    def forward(
        self, 
        inputs):

        # ImgSize: [224x224]
        if self.levels > 0:
            convBlock1 = self.conv_block1(inputs)
        # ImgSize: [224x224]

        if self.levels > 1:
            pool1      = self.maxpool1(convBlock1)
            convBlock2 = self.conv_block2(pool1)
        # ImgSize: [112x112]

        if self.levels > 2:
            pool2      = self.maxpool2(convBlock2)
            convBlock3 = self.conv_block3(pool2)
        # ImgSize: [56x56]

        if self.levels > 3:
            pool3      = self.maxpool3(convBlock3)
            convBlock4 = self.conv_block4(pool3)
        # ImgSize: [28x28]

        if self.levels > 4:
            pool4      = self.maxpool4(convBlock4)
            convBlock5 = self.conv_block5(pool4)
            # ImgSize: [14x14]

            upBlock4 = self.deconv4(convBlock5, output_size=convBlock4.size())
            concat4  = torch.cat([upBlock4, convBlock4],dim=1)
            # ImgSize: [28x28]

        if self.levels > 3:
            upconvBlock4 = self.upconv_block4(concat4)
            upBlock3 = self.deconv3(upconvBlock4, output_size=convBlock3.size())
            concat3  = torch.cat([upBlock3, convBlock3],dim=1)
        # ImgSize: [56x56]

        if self.levels > 2:
            upconvBlock3 = self.upconv_block3(concat3)
            upBlock2 = self.deconv2(upconvBlock3, output_size=convBlock2.size())
            concat2  = torch.cat([upBlock2, convBlock2],dim=1)
        # ImgSize: [112x112]

        if self.levels > 1:
            upconvBlock2 = self.upconv_block2(concat2)
            upBlock1 = self.deconv1(upconvBlock2, output_size=convBlock1.size())
            concat1  = torch.cat([upBlock1, convBlock1],dim=1)
        # ImgSize: [224x224]

        if self.levels > 0:
            upconvBlock1 = self.upconv_block1(concat1)
            out_conv = self.out_conv(upconvBlock1)
            out      = self.out_act(out_conv)
        # ImgSize: [224x224]

        return out


