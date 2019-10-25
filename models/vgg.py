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

    def unpool_forward(
        self,
        input):

        layers = {}

        out    = self.conv1_1(input)
        out    = self.relu1_1(out)
        out    = self.conv1_2(out)
        out1   = self.relu1_2(out)
        
        if self.levels < 2: 
            layers['z'] = out1
            return layers

        layers['out1'] = out1
        out, pool1_idx = self.maxpool1(out1)

        out  = self.conv2_1(out)
        out  = self.relu2_1(out)
        out  = self.conv2_2(out)
        out2 = self.relu2_2(out)

        layers['pool1_idx'] = pool1_idx

        if self.levels < 3: 
            layers['z'] = out2
            return layers
            
        layers['out2'] = out2
        out, pool2_idx = self.maxpool2(out2)

        out  = self.conv3_1(out)
        out  = self.relu3_1(out)
        out  = self.conv3_2(out)
        out  = self.relu3_2(out)
        out  = self.conv3_3(out)
        out3 = self.relu3_3(out)

        layers['pool2_idx'] = pool2_idx  
        
        if self.levels < 4: 
            layers['z'] = out3
            return layers

        layers['out3'] = out3
        out, pool3_idx = self.maxpool3(out3)

        out  = self.conv4_1(out)
        out  = self.relu4_1(out)
        out  = self.conv4_2(out)
        out  = self.relu4_2(out)
        out  = self.conv4_3(out)
        out4 = self.relu4_3(out)

        layers['pool3_idx'] = pool3_idx
        
        if self.levels < 5: 
            layers['z'] = out4
            return layers

        layers['out4'] = out4
        out, pool4_idx = self.maxpool3(out4)

        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)

        layers['z'] = out
        layers['pool4_idx'] = pool4_idx

        return layers

    def deconv_forward(
        self,
        input):

        layers = {}

        out    = self.conv1_1(input)
        out    = self.relu1_1(out)
        out    = self.conv1_2(out)
        out1   = self.relu1_2(out)

        if self.levels < 2: 
            layers['z'] = out1
            return layers

        layers['out1'] = out1
        out, pool1_idx = self.maxpool1(out1)

        out   = self.conv2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        out2  = self.relu2_2(out)

        if self.levels < 3: 
            layers['z'] = out2
            return layers

        layers['out2'] = out2
        out, pool2_idx = self.maxpool2(out2)

        out   = self.conv3_1(out)
        out   = self.relu3_1(out)
        out   = self.conv3_2(out)
        out   = self.relu3_2(out)
        out   = self.conv3_3(out)
        out3  = self.relu3_3(out)

        if self.levels < 4:
            layers['z'] = out3
            return layers

        layers['out3'] = out3
        out, pool3_idx = self.maxpool3(out3)

        out   = self.conv4_1(out)
        out   = self.relu4_1(out)
        out   = self.conv4_2(out)
        out   = self.relu4_2(out)
        out   = self.conv4_3(out)
        out4  = self.relu4_3(out)

        

        if self.levels < 5: 
            layers['z'] = out4
            return layers

        layers['out4'] = out4
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
        mode,
        out_act     = 'tanh',
        use_dropout = False,
        use_bn      = False,
        use_attn    = False,
        prob        = 0.5):

        super(vgg16_decoder, self).__init__()
        self.levels = levels

        # [14x14]
        ch_ini = 512
        if levels > 4:
            
            self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            # Hout = (Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            #
            # output_padding = 0
            # dilation       = 0
            ch_cat = ch_ini*2 # 1024
            hid_ch = ch_ini   # 512
            out_ch = ch_ini  

            if mode == 'unpool': out_ch = int(ch_ini/2) # 256

            self.unconv4 = nn.ConvTranspose2d(in_channels  = ch_ini, 
                                              out_channels = hid_ch, 
                                              kernel_size  = 3, 
                                              stride       = 2, 
                                              padding      = 1)

            self.conv_block4 = self.convBlock(False, prob, use_attn, use_bn, 3, ch_cat, hid_ch, out_ch)

        ch_ini = int(ch_ini/2) # 256
        # [28x28]
        if levels > 3:
            ch_cat = ch_ini*2  # 512
            hid_ch = ch_ini    # 256
            out_ch = ch_ini    

            if mode == 'unpool': out_ch = int(ch_ini/2) # 128

            self.unpool3  = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv3  = nn.ConvTranspose2d(in_channels  = hid_ch*2, 
                                               out_channels = hid_ch, 
                                               kernel_size  = 3, 
                                               stride       = 2, 
                                               padding      = 1)

            self.conv_block3 = self.convBlock(False, prob, use_attn, use_bn, 2, ch_cat, hid_ch, out_ch)

        ch_ini = int(ch_ini/2) # 256
        # [56x56]
        if levels > 2:
            ch_cat = ch_ini*2 # 256
            hid_ch = ch_ini   # 128
            out_ch = ch_ini

            if mode == 'unpool': out_ch = int(ch_ini/2) # 64

            self.unpool2  = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv2  = nn.ConvTranspose2d(in_channels  = hid_ch*2, 
                                               out_channels = hid_ch, 
                                               kernel_size  = 3, 
                                               stride       = 2, 
                                               padding      = 1)

            self.conv_block2 = self.convBlock(False, prob, use_attn, use_bn, 2, ch_cat, hid_ch, out_ch)

        ch_ini = int(ch_ini/2) # 64
        # [112x112]
        if levels > 1:
            ch_cat = ch_ini*2 # 128
            hid_ch = ch_ini   # 64
            out_ch = ch_ini

            self.unpool1  = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv1  = nn.ConvTranspose2d(in_channels  = hid_ch*2, 
                                               out_channels = hid_ch, 
                                               kernel_size  = 3, 
                                               stride       = 2, 
                                               padding      = 1)

            self.conv_block1 = self.convBlock(False, prob, use_attn, use_bn, 2, ch_cat, hid_ch, out_ch)       
            
        # [224x224]
        if levels > 0:
            self.convToCh = nn.Conv2d(ch_ini,3,3,1,1)
            if out_act == 'tanh':
                self.outact = nn.Tanh()
            else:
                self.outact = nn.Sigmoid()


    def convBlock(self, use_drop, drop_prop, use_attn, use_bn, block_size, in_ch, hid_ch, out_ch):
        hid_ch_2nd = hid_ch

        seq = []       
        seq.append(nn.Conv2d(in_ch,hid_ch,3,1,1))
        if use_bn  : seq.append(nn.BatchNorm2d(hid_ch))
        if use_drop: seq.append(nn.Dropout(p=drop_prop, inplace=False))
        seq.append(nn.ReLU(inplace=True))
        
        if block_size == 2:
            hid_ch_2nd = out_ch

        seq.append(nn.Conv2d(hid_ch,hid_ch_2nd,3,1,1))
        if use_bn  : seq.append(nn.BatchNorm2d(intch2nd))
        if use_drop: seq.append(nn.Dropout(p=drop_prop, inplace=False))
        seq.append(nn.ReLU(inplace=True))
        
        if block_size > 2:
            seq.append(nn.Conv2d(hid_ch_2nd,out_ch,3,1,1))
            if use_bn  : seq.append(nn.BatchNorm2d(out_ch))
            if use_drop: seq.append(nn.Dropout(p=drop_prop, inplace=False))
            seq.append(nn.ReLU(inplace=True))   

        return nn.Sequential(*seq)

    def unpool_forward(
        self, 
        layers,
        att_map       = None,
        use_dropout   = False,
        use_attention = False,
        use_bn        = False):

        out = layers['z']
        
        if self.levels > 4:
            enc_out4   = layers['out4']
            pool4_idx  = layers['pool4_idx']
            out_unpool = self.unpool4(out, pool4_idx, output_size=enc_out4.size())
            out_concat = torch.cat((out_unpool,enc_out4), dim=1)
            out        = self.conv_block4(out_concat)

        if self.levels > 3:
            enc_out3   = layers['out3']
            pool3_idx  = layers['pool3_idx']

            out_unpool = self.unpool3(out, pool3_idx, output_size=enc_out3.size())
            out_concat = torch.cat((out_unpool,enc_out3), dim=1)
            out        = self.conv_block3(out_concat)

        if self.levels > 2:
            enc_out2   = layers['out2']
            pool2_idx  = layers['pool2_idx']

            out_unpool = self.unpool2(out, pool2_idx, output_size=enc_out2.size())
            out_concat = torch.cat((out_unpool,enc_out2), dim=1)
            out        = self.conv_block2(out_concat)

        if self.levels > 1:
            enc_out1   = layers['out1']
            pool1_idx  = layers['pool1_idx']

            out_unpool = self.unpool1(out, pool1_idx, output_size=enc_out1.size())
            out_concat = torch.cat((out_unpool,enc_out1), dim=1)
            out        = self.conv_block1(out_concat)

        if self.levels > 0:
            out = self.convToCh(out)
            out = self.outact(out)

        return out   

    def deconv_forward(
        self, 
        layers,
        att_map       = None,
        use_dropout   = False,
        use_attention = False,
        use_bn        = False):

        out = layers['z']

        if self.levels > 4:
            enc_out4   = layers['out4']
            out_unconv = self.unconv4(out, output_size=enc_out4.size())
            out_concat = torch.cat((out_unconv,enc_out4), dim=1)
            out        = self.conv_block4(out_concat)

        if self.levels > 3:
            enc_out3   = layers['out3']
            out_unconv = self.unconv3(out, output_size=enc_out3.size())
            out_concat = torch.cat((out_unconv,enc_out3), dim=1)    
            out        = self.conv_block3(out_concat)   

        if self.levels > 2:
            enc_out2   = layers['out2']
            out_unconv = self.unconv2(out, output_size=enc_out2.size())
            out_concat = torch.cat((out_unconv,enc_out2), dim=1)    
            out        = self.conv_block2(out_concat)   

        if self.levels > 1:
            enc_out1   = layers['out1']
            out_unconv = self.unconv1(out, output_size=enc_out1.size())
            out_concat = torch.cat((out_unconv,enc_out1), dim=1)    
            out        = self.conv_block1(out_concat)   

        if self.levels > 0:
            out = self.convToCh(out)
            out = self.outact(out)

        return out  