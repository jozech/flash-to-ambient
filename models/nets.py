import torch
import torch.nn as nn

from .vgg import vgg_encoder, vgg_decoder

#print(features_list[0].weight.shape)
#print(features_list[2].weight.shape)
#print(features_list[5].weight.shape)
#print(features_list[7].weight.shape)
#print(features_list[10].weight.shape)
#print(features_list[12].weight.shape)
#print(features_list[14].weight.shape)
#print(features_list[16].weight.shape)
#print(features_list[19].weight.shape)

class generator_unpool(nn.Module):
    def __init__(self):
        super(generator_unpool, self).__init__()

        #self.enc1 = vgg_encoder(1)
        #self.dec1 = vgg_decoder(1)
        #self.enc2 = vgg_encoder(2)
        #self.dec2 = vgg_decoder(2)
        #self.enc3 = vgg_encoder(3)
        #self.dec3 = vgg_decoder(3)
        self.enc4 = vgg_encoder(4)
        self.dec4 = vgg_decoder(4)

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


class generator_concat(nn.Module):
    def __init__(self):
        super(generator_concat, self).__init__()

        #self.enc1 = vgg_encoder(1)
        #self.dec1 = vgg_decoder(1)
        #self.enc2 = vgg_encoder(2)
        #self.dec2 = vgg_decoder(2)
        #self.enc3 = vgg_encoder(3)
        #self.dec3 = vgg_decoder(3)
        self.enc4 = vgg_encoder(4)
        self.dec4 = vgg_decoder(4)

    def forward(self, input_imgs):  
        
        out4, out3, out2, out1 = self.enc4.forward_multiple(input_imgs)
        out_img = self.dec4.forward_concat(out4, out3, out2, out1)
            
        return out4, out_img

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