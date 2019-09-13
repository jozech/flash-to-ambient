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
    def __init__(self):
        """
            Discriminator:

            stride  : 2-2-1-1-1
            channels: 64-128-256-512-1
        """
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels= 64,
                               kernel_size = 3,
                               stride      = 2,
                               padding     = 1)
        self.relu1 = nn.LeakyReLU(inplace=True)     
        #[112x112]          
        
        self.conv2 = nn.Conv2d(in_channels = 64,
                               out_channels= 128,
                               kernel_size = 3,
                               stride      = 2,
                               padding     = 1)
        self.bn_2  = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.LeakyReLU(inplace=True)
        #[56x56] 

        self.conv3 = nn.Conv2d(in_channels = 128,
                               out_channels= 256,
                               kernel_size = 3,
                               stride      = 1,
                               padding     = 1)
        self.bn_3  = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.LeakyReLU(inplace=True)
        #[56x56]

        self.conv4 = nn.Conv2d(in_channels = 256,
                               out_channels= 512,
                               kernel_size = 3,
                               stride      = 1,
                               padding     = 1)
        self.bn_4  = nn.BatchNorm2d(num_features=512)
        self.relu4 = nn.LeakyReLU(inplace=True)
        #[56x56]

        self.conv5 = nn.Conv2d(in_channels = 512,
                               out_channels= 1,
                               kernel_size = 3,
                               stride      = 1,
                               padding     = 1)
        #[56x56]
    
    def forward(self, input_pair):
        out = self.conv1(input_pair)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn_2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn_3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.bn_4(out)
        out = self.relu4(out)

        out = self.conv5(out)
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


class UNet:
    def __init__(self, img_ch, init_ch):
        super(UNet, self).__init__()
        self.init_ch     = init_ch
        self.conv_block1 = self.conv_block(img_ch, init_ch)
        self.maxpool1    = nn.MaxPool2d(2,2)
        self.conv_block2 = self.conv_block(init_ch, init_ch*2)
        self.maxpool2    = nn.MaxPool2d(2,2)
        self.conv_block3 = self.conv_block(init_ch*2, init_ch*4)
        self.maxpool3    = nn.MaxPool2d(2,2)
        self.conv_block4 = self.conv_block(init_ch*4, init_ch*8)
        self.maxpool4    = nn.MaxPool2d(2,2)
        self.conv_block5 = self.conv_block(init_ch*8, init_ch*16)

        self.deconv4       = nn.ConvTranspose2d(init_ch*16, init_ch*8, 2, stride=2)
        self.upconv_block4 = self.conv_block(init_ch*16, init_ch*8)
        self.deconv3       = nn.ConvTranspose2d(init_ch*8, init_ch*4, 2, stride=2)
        self.upconv_block4 = self.conv_block(init_ch*8, init_ch*4)
        self.deconv2       = nn.ConvTranspose2d(init_ch*4, init_ch*2, 2, stride=2)
        self.upconv_block2 = self.conv_block(init_ch*4, init_ch*2)     
        self.deconv1       = nn.ConvTranspose2d(init_ch*2, init_ch, 2, stride=2)
        self.upconv_block1 = self.conv_block(init_ch*2, init_ch)      

        self.out_conv      = nn.Conv2d(init_ch, img_ch, 1, padding=1)

    def conv_block(self, in_ch, out_ch):
        return  nn.Sequential(
                    nn.Conv2d(in_ch , out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )

    def forward(
        self, 
        inputs):

        block1 = self.conv_block1(inputs)
        pool1  = self.maxpool1(block1)
        block2 = self.conv_block2(pool1)
        pool2  = self.maxpool2(block2)
        block3 = self.conv_block3(pool2)
        pool3  = self.maxpool3(block3)
        block4 = self.conv_block1(pool3)
        pool4  = self.maxpool4(block4)
        block5 = self.conv_block1(pool4)

        upblock4 = self.deconv4(block5)
        concat4  = torch.cat([pool4, upblock4])
        block4   = self.upconv_block4(concat4)

        upblock3 = self.deconv3(block4)
        concat3  = torch.cat([pool3, upblock3])
        block3   = self.upconv_block3(concat3)

        upblock2 = self.deconv2(block3)
        concat2  = torch.cat([pool2, upblock2])
        block2   = self.upconv_block2(concat2)

        upblock1 = self.deconv1(block2)
        concat1  = torch.cat([pool1, upblock1])
        block1   = self.upconv_block5(concat1)

        out  = self.out_conv(block1)

        return out


