import torch
import torch.nn as nn

class vgg16_encoder(nn.Module):
    def __init__(
        self, 
        levels):

        super(vgg16_encoder, self).__init__()
        self.levels   = levels

        self.conv1_1 = nn.Conv2d(in_channels = 3,
                                 out_channels= 64,
                                 kernel_size = 3,
                                 stride      = 1,
                                 padding     = 1)
        self.relu1_1 = nn.ReLU(inplace=True)    

        self.conv1_2 = nn.Conv2d(in_channels = 64,
                                 out_channels= 64,
                                 kernel_size = 3,
                                 stride      = 1,
                                 padding     = 1)

        self.relu1_2 = nn.ReLU(inplace=True)    
        # [224x224]

        if levels < 2: return

        self.maxpool1= nn.MaxPool2d(kernel_size=2, 
                                    stride=2,
                                    padding=0,
                                    dilation=1,
                                    return_indices=True, 
                                    ceil_mode=False)

        self.conv2_1 = nn.Conv2d(64,128,3,1,1)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.ReLU(inplace=True)
        # [112x112]

        if levels < 3: return

        self.maxpool2 = nn.MaxPool2d(kernel_size  = 2, 
                                     stride       = 2,
                                     return_indices=True)
        
        self.conv3_1  = nn.Conv2d(128,256,3,1,1)
        self.relu3_1  = nn.ReLU(inplace=True)

        self.conv3_2  = nn.Conv2d(256,256,3,1,1)
        self.relu3_2  = nn.ReLU(inplace=True)

        self.conv3_3  = nn.Conv2d(256,256,3,1,1)
        self.relu3_3  = nn.ReLU(inplace=True)
        # [56x56]

        if levels < 4: return

        self.maxpool3 = nn.MaxPool2d(kernel_size   = 2, 
                                     stride        = 2,
                                     return_indices= True)

        self.conv4_1  = nn.Conv2d(256,512,3,1,1)
        self.relu4_1  = nn.ReLU(inplace=True)

        self.conv4_2  = nn.Conv2d(512,512,3,1,1)
        self.relu4_2  = nn.ReLU(inplace=True)

        self.conv4_3  = nn.Conv2d(512,512,3,1,1)
        self.relu4_3  = nn.ReLU(inplace=True)
        # [28x28]

        if levels < 5: return

        self.maxpool4 = nn.MaxPool2d(kernel_size   = 2, 
                                     stride        = 2,
                                     return_indices= True)

        self.conv5_1  = nn.Conv2d(512,512,3,1,1)
        self.relu5_1  = nn.ReLU(inplace=True)

        self.conv5_2  = nn.Conv2d(512,512,3,1,1)
        self.relu5_2  = nn.ReLU(inplace=True)

        self.conv5_3  = nn.Conv2d(512,512,3,1,1)
        self.relu5_3  = nn.ReLU(inplace=True)
        # [14x14]

    def forward(
        self,
        input):

        out   = self.conv1_1(input)
        out   = self.relu1_1(out)
        out   = self.conv1_2(out)
        pool1 = self.relu1_2(out)
        
        layers = {'pool1': pool1}
            
        if self.levels < 2: return layers

        out, pool1_idx = self.maxpool1(pool1)
        out   = self.conv2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        layers['pool2'] = pool2
        layers['pool1_idx']  = pool1_idx
        layers['pool1_size'] = pool1.size()

        if self.levels < 3: return layers

        out, pool2_idx = self.maxpool2(pool2)

        out   = self.conv3_1(out)
        out   = self.relu3_1(out)
        out   = self.conv3_2(out)
        out   = self.relu3_2(out)
        out   = self.conv3_3(out)
        pool3 = self.relu3_3(out)

        layers['pool3'] = pool3
        layers['pool2_idx']  = pool2_idx
        layers['pool2_size'] = pool2.size()      
        
        if self.levels < 4: return layers

        out, pool3_idx = self.maxpool3(pool3)

        out   = self.conv4_1(out)
        out   = self.relu4_1(out)
        out   = self.conv4_2(out)
        out   = self.relu4_2(out)
        out   = self.conv4_3(out)
        pool4 = self.relu4_3(out)

        layers['pool4'] = pool4
        layers['pool3_idx']  = pool3_idx
        layers['pool3_size'] = pool3.size()    
        
        if self.levels < 5: return layers

        out, pool4_idx = self.maxpool3(pool4)

        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)

        layers['z'] = out
        layers['pool4_idx']  = pool4_idx
        layers['pool4_size'] = pool4.size()

        return layers

    def forward_multiple(
        self,
        input):

        out   = self.conv1_1(input)
        out   = self.relu1_1(out)
        out   = self.conv1_2(out)
        out1 = self.relu1_2(out)
        
        layers = {'out1': out1}
    
        if self.levels < 2: 
            layers['z'] = out1
            return layers

        out, pool1_idx = self.maxpool1(out1)
        out   = self.conv2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        out2  = self.relu2_2(out)

        layers['out2'] = out2

        if self.levels < 3: 
            layers['z'] = out2
            return layers

        out, pool2_idx = self.maxpool2(out2)

        out   = self.conv3_1(out)
        out   = self.relu3_1(out)
        out   = self.conv3_2(out)
        out   = self.relu3_2(out)
        out   = self.conv3_3(out)
        out3  = self.relu3_3(out)
        
        layers['out3'] = out3

        if self.levels < 4:
            layers['z'] = out3
            return layers

        out, pool3_idx = self.maxpool3(out3)

        out   = self.conv4_1(out)
        out   = self.relu4_1(out)
        out   = self.conv4_2(out)
        out   = self.relu4_2(out)
        out   = self.conv4_3(out)
        out4  = self.relu4_3(out)

        layers['out4'] = out4

        if self.levels < 5: 
            layers['z'] = out4
            return layers

        out, pool4_idx = self.maxpool3(out4)

        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)

        layers['z'] = out

        return layers

class vgg16_decoder(nn.Module):
    def __init__(
        self, 
        levels,
        use_dropout = False,
        use_bn      = False,
        prob        = 0.5):

        super(vgg16_decoder, self).__init__()
        self.levels = levels

        # [14x14]
        if levels > 4:
            self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv4 = nn.ConvTranspose2d(in_channels =512, 
                                              out_channels=512, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')
            self.bn4_cat1 = nn.BatchNorm2d(512)
            self.bn4_cat2 = nn.BatchNorm2d(512)

            self.conv4_3 = nn.Conv2d(1024,512,3,1,1)
            self.bn4_3   = nn.BatchNorm2d(512)
            self.drop4_3 = nn.Dropout(p=prob, inplace=False)
            self.relu4_3 = nn.ReLU(inplace=True)
            self.conv4_2 = nn.Conv2d(512,512,3,1,1)
            self.bn4_2   = nn.BatchNorm2d(512)
            self.drop4_2 = nn.Dropout(p=prob, inplace=False)
            self.relu4_2 = nn.ReLU(inplace=True)
            self.conv4_1 = nn.Conv2d(512,512,3,1,1)
            self.bn4_1   = nn.BatchNorm2d(512)
            self.drop4_1 = nn.Dropout(p=prob, inplace=False)
            self.relu4_1 = nn.ReLU(inplace=True)

        # [28x28]
        if levels > 3:
            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv3 = nn.ConvTranspose2d(in_channels =512, 
                                              out_channels=256, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')
            self.bn3_cat1 = nn.BatchNorm2d(256)
            self.bn3_cat2 = nn.BatchNorm2d(256)

            self.conv3_3 = nn.Conv2d(512,256,3,1,1)
            self.bn3_3   = nn.BatchNorm2d(256)
            self.drop3_3 = nn.Dropout(p=prob, inplace=False)
            self.relu3_3 = nn.ReLU(inplace=True)
            self.conv3_2 = nn.Conv2d(256,256,3,1,1)
            self.bn3_2   = nn.BatchNorm2d(256)
            self.drop3_2 = nn.Dropout(p=prob, inplace=False)
            self.relu3_2 = nn.ReLU(inplace=True)
            self.conv3_1 = nn.Conv2d(256,256,3,1,1)
            self.bn3_1   = nn.BatchNorm2d(256)
            self.drop3_1 = nn.Dropout(p=prob, inplace=False)
            self.relu3_1 = nn.ReLU(inplace=True)

        # [56x56]
        if levels > 2:
            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv2 = nn.ConvTranspose2d(in_channels =256, 
                                              out_channels=128, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')
            self.bn2_cat1 = nn.BatchNorm2d(128)
            self.bn2_cat2 = nn.BatchNorm2d(128)

            self.conv2_2 = nn.Conv2d(256,128,3,1,1)
            self.bn2_2   = nn.BatchNorm2d(128)
            self.relu2_2 = nn.ReLU(inplace=True)
            self.conv2_1 = nn.Conv2d(128,128,3,1,1)
            self.bn2_1   = nn.BatchNorm2d(128)
            self.relu2_1 = nn.ReLU(inplace=True)

        # [112x112]
        if levels > 1:
            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv1 = nn.ConvTranspose2d(in_channels =128, 
                                              out_channels=64, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')
            self.bn1_cat1 = nn.BatchNorm2d(64)
            self.bn1_cat2 = nn.BatchNorm2d(64)

            self.conv1_2 = nn.Conv2d(128,64,3,1,1)
            self.bn1_2   = nn.BatchNorm2d(64)
            self.relu1_2 = nn.ReLU(inplace=True)
            self.conv1_1 = nn.Conv2d(64,3,3,1,1)

        # [224x224]
        if levels > 0:
            self.tanh0 = nn.Tanh()

    def forward_unpool(
        self, 
        layers,
        use_dropout = False,
        use_bn      = False):

        out = layers['z']
        
        if self.levels > 4:
            pool4_size = layers['pool4_size']
            pool4_idx  = layers['pool4_idx']
            pool4      = layers['pool4']

            out = self.unpool4(out, pool4_idx, output_size=pool4_size)  # [None, 512,28,28]
            if use_dropout:
                out   = self.bn4_cat1(out)
                pool4 = self.bn4_cat2(pool4)
            out = torch.cat((out,pool4), dim=1)                         # [None,1024,28,28]
            
            out = self.conv4_3(out)
            if use_bn:      out = self.bn4_3(out)
            if use_dropout: out = self.drop4_3(out)
            out = self.relu4_3(out)

            out = self.conv4_2(out)
            if use_bn:      out = self.bn4_2(out)
            if use_dropout: out = self.drop4_2(out)
            out = self.relu4_2(out)
            
            out = self.conv4_1(out)
            if use_bn:      out = self.bn4_1(out)
            if use_dropout: out = self.drop4_1(out)
            out = self.relu4_1(out)

        if self.levels > 3:
            pool3_size = layers['pool3_size']
            pool3_idx  = layers['pool3_idx']
            pool3      = layers['pool3']

            out = self.unpool3(out, pool3_idx, output_size=pool3_size)
            if use_bn:
                out   = self.bn3_cat1(out)
                pool3 = self.bn3_cat2(pool3)
            out = torch.cat((out,pool3), dim=1)

            out = self.conv3_3(out)
            if use_bn:      out = self.bn3_3(out)
            if use_dropout: out = self.drop3_3(out)
            out = self.relu3_3(out)

            out = self.conv3_2(out)
            if use_bn:      out = self.bn3_2(out)
            if use_dropout: out = self.drop3_2(out)
            out = self.relu3_2(out)
            
            out = self.conv3_1(out)
            if use_bn: out = self.bn3_1(out)
            if use_dropout: out = self.drop3_1(out)
            out = self.relu3_1(out)

        if self.levels > 2:
            pool2_size = layers['pool2_size']
            pool2_idx  = layers['pool2_idx']
            pool2      = layers['pool2']

            out = self.unpool2(out, pool2_idx, output_size=pool2_size)
            if use_bn:
                out   = self.bn2_cat1(out)
                pool2 = self.bn2_cat2(pool2)
            out = torch.cat((out,pool2), dim=1)

            out = self.conv2_2(out)
            if use_bn: out = self.bn2_2(out)
            out = self.relu2_2(out)

            out = self.conv2_1(out)
            if use_bn: out = self.bn2_1(out)
            out = self.relu2_1(out)

        if self.levels > 1:
            pool1_size = layers['pool1_size']
            pool1_idx  = layers['pool1_idx']
            pool1      = layers['pool1']

            out   = self.unpool1(out, pool1_idx, output_size=pool1_size)
            if use_bn:
                out   = self.bn1_cat1(out)
                pool1 = self.bn1_cat2(pool1)
            out = torch.cat((out,pool1), dim=1)

            out = self.conv1_2(out)
            if use_bn: out = self.bn1_2(out)
            out = self.relu1_2(out)

            out = self.conv1_1(out)

        if self.levels > 0:
            out = self.tanh0(out)
        return out

    def forward_deconv(
        self, 
        layers,
        gray_img     = None,
        use_dropout  = False,
        use_attention= False,
        use_bn       = False):

        out = layers['z']

        att_map = 1.0 - (gray_img * 0.5 + 0.5)

        if self.levels > 4:
            out4    = layers['out4']
            if use_attention:
                att_map4 = att_map.resize_(out4.size())
                out4     = torch.mul(att_map4, out4)

            out   = self.unconv4(out, output_size=out4.size())
            if use_bn:
                out   = self.bn4_cat1(out)
                out4 = self.bn4_cat2(out4)
            out   = torch.cat((out,out4), dim=1)

            out = self.conv4_3(out)
            if use_bn:      out = self.bn4_3(out)
            if use_dropout: out = self.drop4_3(out)
            out = self.relu4_3(out)
            
            out = self.conv4_2(out)
            if use_bn: out = self.bn4_2(out)
            if use_dropout: out = self.drop4_2(out)
            out = self.relu4_2(out)

            out = self.conv4_1(out)
            if use_bn:      out = self.bn4_1(out)
            if use_dropout: out = self.drop4_1(out)
            out = self.relu4_1(out)

        if self.levels > 3:
            out3 = layers['out3']
            if use_attention:
                att_map3 = att_map.resize_(out3.size())
                out3     = torch.mul(att_map3, out3)
            out   = self.unconv3(out, output_size=out3.size())
            if use_bn:
                out  = self.bn3_cat1(out)
                out3 = self.bn3_cat2(out3)
            out   = torch.cat((out,out3), dim=1)

            out = self.conv3_3(out)
            if use_bn:      out = self.bn3_3(out)
            if use_dropout: out = self.drop3_3(out)
            out = self.relu3_3(out)
            
            out = self.conv3_2(out)
            if use_bn:      out = self.bn3_2(out)
            if use_dropout: out = self.drop3_2(out)
            out = self.relu3_2(out)
            
            out = self.conv3_1(out)
            if use_bn:      out = self.bn3_1(out)
            if use_dropout: out = self.drop3_1(out)
            out = self.relu3_1(out)

        if self.levels > 2:
            out2 = layers['out2']
            if use_attention:
                att_map2 = att_map.resize_(out2.size())
                out2 = torch.mul(att_map2, out2)
            out   = self.unconv2(out, output_size=out2.size())
            if use_bn:
                out  = self.bn2_cat1(out)
                out2 = self.bn2_cat2(out2)
            out   = torch.cat((out,out2), dim=1)

            out = self.conv2_2(out)
            if use_bn:      out = self.bn2_2(out)
            if use_dropout: out = self.relu2_2(out)
            
            out = self.conv2_1(out)
            if use_bn:out = self.bn2_1(out)
            out = self.relu2_1(out)

        if self.levels > 1:
            out1 = layers['out1']
            if use_attention:
                att_map1 = att_map.resize_(out1.size())
                out1 = torch.mul(att_map1, out1)
            out   = self.unconv1(out, output_size=out1.size())
            if use_bn:
                out  = self.bn1_cat1(out)
                out1 = self.bn1_cat2(out1)
            out = torch.cat((out,out1), dim=1)

            out = self.conv1_2(out)
            if use_bn:  out = bn1_2(out)
            out = self.relu1_2(out)

            out = self.conv1_1(out)

        if self.levels > 0:
            out = self.tanh0(out)

        return out