import torch
import torch.nn as nn

from .vgg import vgg19_encoder, vgg19_decoder
from .vgg import vgg16_encoder, vgg16_decoder

class vgg19_generator_unpool(nn.Module):
    def __init__(self):
        super(generator_unpool, self).__init__()
        self.enc4 = vgg19_encoder(4)
        self.dec4 = vgg19_decoder(4)

    def forward(self, input_imgs):  
        
        iF4, ipool_idx, ipool1, ipool_idx2, ipool2, ipool_idx3, ipool3 = self.enc4(input_imgs)
        out = self.dec4(iF4, ipool_idx, ipool1, ipool_idx2, ipool2, ipool_idx3, ipool3)
            
        return iF4, out

    def set_vgg_as_encoder(self):
        from torchvision import models
        
        vgg19 = models.vgg19(pretrained=True, progress=True)
        features_list = list(vgg19.features)

        with torch.no_grad():
            self.enc4.conv1_1.weight.copy_(features_list[0].weight)
            self.enc4.conv1_1.bias.copy_(features_list[0].bias)

            self.enc4.conv1_2.weight.copy_(features_list[2].weight)
            self.enc4.conv1_2.bias.copy_(features_list[2].bias)

            self.enc4.conv2_1.weight.copy_(features_list[5].weight)
            self.enc4.conv2_1.bias.copy_(features_list[5].bias)

            self.enc4.conv2_2.weight.copy_(features_list[7].weight)
            self.enc4.conv2_2.bias.copy_(features_list[7].bias)

            self.enc4.conv3_1.weight.copy_(features_list[10].weight)
            self.enc4.conv3_1.bias.copy_(features_list[10].bias)

            self.enc4.conv3_2.weight.copy_(features_list[12].weight)
            self.enc4.conv3_2.bias.copy_(features_list[12].bias)

            self.enc4.conv3_3.weight.copy_(features_list[14].weight)
            self.enc4.conv3_3.bias.copy_(features_list[14].bias)

            self.enc4.conv3_4.weight.copy_(features_list[16].weight)
            self.enc4.conv3_4.bias.copy_(features_list[16].bias)

            self.enc4.conv4_1.weight.copy_(features_list[19].weight)
            self.enc4.conv4_1.bias.copy_(features_list[19].bias)


class vgg19_generator_concat(nn.Module):
    def __init__(self):
        super(generator_concat, self).__init__()

        self.enc4 = vgg19_encoder(4)
        self.dec4 = vgg19_decoder(4)

    def forward(self, input_imgs):  
        
        out4, out3, out2, out1 = self.enc4.forward_multiple(input_imgs)
        out_img = self.dec4.forward_concat(out4, out3, out2, out1)
            
        return out4, out_img

    def set_vgg_as_encoder(self):
        from torchvision import models
        
        vgg16 = models.vgg16(pretrained=True, progress=True)
        features_list = list(vgg16.features)

        for layer in vgg16.features:
            print(layer)
        with torch.no_grad():
            self.enc4.conv1_1.weight.copy_(features_list[0].weight)
            self.enc4.conv1_1.bias.copy_(features_list[0].bias)

            self.enc4.conv1_2.weight.copy_(features_list[2].weight)
            self.enc4.conv1_2.bias.copy_(features_list[2].bias)

            self.enc4.conv2_1.weight.copy_(features_list[5].weight)
            self.enc4.conv2_1.bias.copy_(features_list[5].bias)

            self.enc4.conv2_2.weight.copy_(features_list[7].weight)
            self.enc4.conv2_2.bias.copy_(features_list[7].bias)

            self.enc4.conv3_1.weight.copy_(features_list[10].weight)
            self.enc4.conv3_1.bias.copy_(features_list[10].bias)

            self.enc4.conv3_2.weight.copy_(features_list[12].weight)
            self.enc4.conv3_2.bias.copy_(features_list[12].bias)

            self.enc4.conv3_3.weight.copy_(features_list[14].weight)
            self.enc4.conv3_3.bias.copy_(features_list[14].bias)

            self.enc4.conv3_4.weight.copy_(features_list[16].weight)
            self.enc4.conv3_4.bias.copy_(features_list[16].bias)

            self.enc4.conv4_1.weight.copy_(features_list[19].weight)
            self.enc4.conv4_1.bias.copy_(features_list[19].bias)


class vgg16_generator_unpool(nn.Module):
    def __init__(self, prob):
        super(vgg16_generator_unpool, self).__init__()

        self.enc5 = vgg16_encoder(5)
        self.dec5 = vgg16_decoder(5, prob)

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
            self.enc5.conv1_1.weight.copy_(features_list[0].weight)
            self.enc5.conv1_1.bias.copy_(features_list[0].bias)
            self.enc5.conv1_2.weight.copy_(features_list[2].weight)
            self.enc5.conv1_2.bias.copy_(features_list[2].bias)
            
            # [112x112]
            self.enc5.conv2_1.weight.copy_(features_list[5].weight)
            self.enc5.conv2_1.bias.copy_(features_list[5].bias)
            self.enc5.conv2_2.weight.copy_(features_list[7].weight)
            self.enc5.conv2_2.bias.copy_(features_list[7].bias)

            # [56x56]
            self.enc5.conv3_1.weight.copy_(features_list[10].weight)
            self.enc5.conv3_1.bias.copy_(features_list[10].bias)
            self.enc5.conv3_2.weight.copy_(features_list[12].weight)
            self.enc5.conv3_2.bias.copy_(features_list[12].bias)
            self.enc5.conv3_3.weight.copy_(features_list[14].weight)
            self.enc5.conv3_3.bias.copy_(features_list[14].bias)

            # [28x28]
            self.enc5.conv4_1.weight.copy_(features_list[17].weight)
            self.enc5.conv4_1.bias.copy_(features_list[17].bias)
            self.enc5.conv4_2.weight.copy_(features_list[19].weight)
            self.enc5.conv4_2.bias.copy_(features_list[19].bias)
            self.enc5.conv4_3.weight.copy_(features_list[21].weight)
            self.enc5.conv4_3.bias.copy_(features_list[21].bias)

            # [14x14]
            self.enc5.conv5_1.weight.copy_(features_list[24].weight)
            self.enc5.conv5_1.bias.copy_(features_list[24].bias)
            self.enc5.conv5_2.weight.copy_(features_list[26].weight)
            self.enc5.conv5_2.bias.copy_(features_list[26].bias)
            self.enc5.conv5_3.weight.copy_(features_list[28].weight)
            self.enc5.conv5_3.bias.copy_(features_list[28].bias)



class vgg16_generator_deconv(nn.Module):
    def __init__(self, prob):
        super(vgg16_generator_deconv, self).__init__()

        self.enc5 = vgg16_encoder(5)
        self.dec5 = vgg16_decoder(5, prob)

    def forward(self, input_imgs):  

        z, pool4, pool3, pool2, pool1 = self.enc5.forward_multiple(input_imgs)
        out_img = self.dec5.forward_concat(z, pool4, pool3, pool2, pool1)
            
        return z, out_img

    def set_vgg_as_encoder(self):
        from torchvision import models
        
        vgg16 = models.vgg16(pretrained=True, progress=True)
        features_list = list(vgg16.features)

        with torch.no_grad():
            # [224x224]
            self.enc5.conv1_1.weight.copy_(features_list[0].weight)
            self.enc5.conv1_1.bias.copy_(features_list[0].bias)
            self.enc5.conv1_2.weight.copy_(features_list[2].weight)
            self.enc5.conv1_2.bias.copy_(features_list[2].bias)
            
            # [112x112]
            self.enc5.conv2_1.weight.copy_(features_list[5].weight)
            self.enc5.conv2_1.bias.copy_(features_list[5].bias)
            self.enc5.conv2_2.weight.copy_(features_list[7].weight)
            self.enc5.conv2_2.bias.copy_(features_list[7].bias)

            # [56x56]
            self.enc5.conv3_1.weight.copy_(features_list[10].weight)
            self.enc5.conv3_1.bias.copy_(features_list[10].bias)
            self.enc5.conv3_2.weight.copy_(features_list[12].weight)
            self.enc5.conv3_2.bias.copy_(features_list[12].bias)
            self.enc5.conv3_3.weight.copy_(features_list[14].weight)
            self.enc5.conv3_3.bias.copy_(features_list[14].bias)

            # [28x28]
            self.enc5.conv4_1.weight.copy_(features_list[17].weight)
            self.enc5.conv4_1.bias.copy_(features_list[17].bias)
            self.enc5.conv4_2.weight.copy_(features_list[19].weight)
            self.enc5.conv4_2.bias.copy_(features_list[19].bias)
            self.enc5.conv4_3.weight.copy_(features_list[21].weight)
            self.enc5.conv4_3.bias.copy_(features_list[21].bias)

            # [14x14]
            self.enc5.conv5_1.weight.copy_(features_list[24].weight)
            self.enc5.conv5_1.bias.copy_(features_list[24].bias)
            self.enc5.conv5_2.weight.copy_(features_list[26].weight)
            self.enc5.conv5_2.bias.copy_(features_list[26].bias)
            self.enc5.conv5_3.weight.copy_(features_list[28].weight)
            self.enc5.conv5_3.bias.copy_(features_list[28].bias)

#class discriminator(nn.Module):