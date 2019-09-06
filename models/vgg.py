import torch
import torch.nn as nn

class vgg19_encoder(nn.Module):
    def __init__(self, level):
        super(vgg19_encoder, self).__init__()
        self.level = level
        self.pad1_1 = nn.ReflectionPad2d((1,1,1,1)) # [left, right, top, bottom]

        self.conv1_1 = nn.Conv2d(in_channels = 3,
                                 out_channels= 64,
                                 kernel_size = 3,
                                 stride      = 1,
                                 padding     = 0)

        self.relu1_1 = nn.ReLU(inplace=True)    

        if level < 2: return

        self.pad1_2  = nn.ReflectionPad2d((1,1,1,1))
        self.conv1_2 = nn.Conv2d(64,64,3,1,0)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.maxpool1= nn.MaxPool2d(kernel_size=2, 
                                    stride=2,
                                    padding=0,
                                    dilation=1,
                                    return_indices=True, 
                                    ceil_mode=False)

        self.pad2_1  = nn.ReflectionPad2d((1,1,1,1))
        self.conv2_1 = nn.Conv2d(64,128,3,1,0)
        self.relu2_1 = nn.ReLU(inplace=True)

        if level < 3: return

        self.pad2_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv2_2= nn.Conv2d(128,128,3,1,0)
        self.relu2_2= nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size  =2, 
                                    stride        =2,
                                    return_indices=True)
        
        self.pad3_1   = nn.ReflectionPad2d((1,1,1,1))
        self.conv3_1  = nn.Conv2d(128,256,3,1,0)
        self.relu3_1  = nn.ReLU(inplace=True)

        if level < 4: return
        
        self.pad3_2   = nn.ReflectionPad2d((1,1,1,1))
        self.conv3_2  = nn.Conv2d(256,256,3,1,0)
        self.relu3_2  = nn.ReLU(inplace=True) 

        self.pad3_3   = nn.ReflectionPad2d((1,1,1,1))
        self.conv3_3  = nn.Conv2d(256,256,3,1,0)
        self.relu3_3  = nn.ReLU(inplace=True)

        self.pad3_4   = nn.ReflectionPad2d((1,1,1,1))
        self.conv3_4  = nn.Conv2d(256,256,3,1,0)
        self.relu3_4  = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size  =2, 
                                    stride        =2,
                                    return_indices=True)

        self.pad4_1   = nn.ReflectionPad2d((1,1,1,1))
        self.conv4_1  = nn.Conv2d(256,512,3,1,0)
        self.relu4_1  = nn.ReLU(inplace=True)    # [28x28]

    def forward(self,x):
        out = self.pad1_1(x)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        if self.level < 2: 
            return out

        out   = self.pad1_2(out)
        out   = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        if self.level < 3: 
            return out, pool1_idx, poo1.size()

        out   = self.pad2_2(out)
        out   = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        if self.level < 4:
            return out, pool1_idx, pool1.size(), pool2_idx, pool2.size()

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)

        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, pool1_idx, pool1.size(), pool2_idx, pool2.size(), pool3_idx, pool3.size()

    def forward_multiple(self, x):
        out = self.pad1_1(x)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        if self.level < 2: 
            return out
        
        out1  = out
        out   = self.pad1_2(out)
        out   = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_id = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        if self.level < 3: 
            return out, out1
        
        out2  = out
        out   = self.pad2_2(out)
        out   = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_id = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        if self.level < 4:
            return out, out2, out1

        out3 = out

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, out3, out2, out1

class vgg19_decoder(nn.Module):
    def __init__(self, level):
        super(vgg19_decoder, self).__init__()
        self.level = level

        if level > 3:
            self.pad4_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv4_1 = nn.Conv2d(512,256,3,1,0)
            self.relu4_1 = nn.ReLU(inplace=True)     #[28,28]

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv3 = nn.ConvTranspose2d(in_channels=256, 
                                              out_channels=256, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')

            self.pad3_4  = nn.ReflectionPad2d((1,1,1,1))
            self.conv3_4 = nn.Conv2d(512,256,3,1,0)
            self.relu3_4 = nn.ReLU(inplace=True)

            self.pad3_3  = nn.ReflectionPad2d((1,1,1,1))
            self.conv3_3 = nn.Conv2d(256,256,3,1,0)
            self.relu3_3 = nn.ReLU(inplace=True)

            self.pad3_2  = nn.ReflectionPad2d((1,1,1,1))
            self.conv3_2 = nn.Conv2d(256,256,3,1,0)
            self.relu3_2 = nn.ReLU(inplace=True)


        if level > 2:
            self.pad3_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv3_1 = nn.Conv2d(256,128,3,1,0)
            self.relu3_1 = nn.ReLU(inplace=True)

            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv2 = nn.ConvTranspose2d(in_channels=128, 
                                              out_channels=128, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')

            self.pad2_2  = nn.ReflectionPad2d((1,1,1,1))
            self.conv2_2 = nn.Conv2d(256,128,3,1,0)
            self.relu2_2 = nn.ReLU(inplace=True)

        if level > 1:
            self.pad2_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu2_1 = nn.ReLU(inplace=True)

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv1 = nn.ConvTranspose2d(in_channels=64, 
                                              out_channels=64, 
                                              kernel_size=3, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1, 
                                              groups=1, 
                                              bias=True, 
                                              dilation=1, 
                                              padding_mode='zeros')

            self.pad1_2  = nn.ReflectionPad2d((1,1,1,1))
            self.conv1_2 = nn.Conv2d(128,64,3,1,0)
            self.relu1_2 = nn.ReLU(inplace=True)

        if level > 0:
            self.pad1_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv1_1 = nn.Conv2d(64,3,3,1,0)
            self.tanh1   = nn.Tanh()

    def forward(self, 
                x, 
                pool1_idx =None, 
                pool1_size=None,
                pool2_idx =None,
                pool2_size=None,
                pool3_idx =None,
                pool3_size=None
                ):

        out = x
        if self.level > 3:
            out = self.pad4_1(out)
            out = self.conv4_1(out)
            out = self.relu4_1(out)
            #print(out.shape)

            out = self.unpool3(out, pool3_idx, output_size=pool3_size)
            #print(out.shape)

            out = self.pad3_4(out)
            out = self.conv3_4(out)
            out = self.relu3_4(out)
            #print(out.shape)

            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)
            #print(out.shape)

            out = self.pad3_2(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)
            #print(out.shape)

        if self.level > 2:
            out = self.pad3_1(out)
            out = self.conv3_1(out)
            out = self.relu3_1(out)
            #print(out.shape)

            out = self.unpool2(out, pool2_idx, output_size=pool2_size)
            #print(out.shape)

            out = self.pad2_2(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)
            #print(out.shape)

        if self.level > 1:
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out = self.relu2_1(out)
            #print(out.shape)

            out = self.unpool1(out, pool1_idx, output_size=pool1_size)
            #print(out.shape)

            out = self.pad1_2(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)
            #print(out.shape)

        if self.level > 0:
            out = self.pad1_1(out)
            out = self.conv1_1(out)
            out = self.tanh1(out)
            #print(out.shape)

        return out

    def forward_concat(self,
                out4, 
                out3,
                out2,
                out1
                ):

        out = out4

        if self.level > 3:
            out = self.pad4_1(out)
            out = self.conv4_1(out)
            out = self.relu4_1(out)
            #print(out.shape)
            
            out = self.unconv3(out, output_size=out3.size())
            out = torch.cat((out,out3), dim=1)
            #print(out.shape, 'cat')

            out = self.pad3_4(out)
            out = self.conv3_4(out)
            out = self.relu3_4(out)
            #print(out.shape)

            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)
            #print(out.shape)

            out = self.pad3_2(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)
            #print(out.shape)

        if self.level > 2:
            out = self.pad3_1(out)
            out = self.conv3_1(out)
            out = self.relu3_1(out)
            #print(out.shape)

            out = self.unconv2(out, output_size=out2.size())
            out = torch.cat((out,out2), dim=1)
            #print(out.shape, 'cat')

            out = self.pad2_2(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)
            #print(out.shape)

        if self.level > 1:
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out = self.relu2_1(out)
            #print(out.shape)

            out = self.unconv1(out, output_size=out1.size())
            out = torch.cat((out,out1), dim=1)
            #print(out.shape, 'cat')

            out = self.pad1_2(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)
            #print(out.shape)

        if self.level > 0:
            out = self.pad1_1(out)
            out = self.conv1_1(out)
            out = self.tanh1(out)
            #print(out.shape)

        return out


class vgg16_encoder(nn.Module):
    def __init__(self, level):
        super(vgg16_encoder, self).__init__()
        self.level   = level

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

        if level < 2: return

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

        if level < 3: return

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

        if level < 4: return

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

        if level < 5: return

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

    def forward(self,x):
        out   = self.conv1_1(x)
        out   = self.relu1_1(out)
        out   = self.conv1_2(out)
        pool1 = self.relu1_2(out)
        
        layers = {'pool1': pool1}
            
        if self.level < 2: return layers

        out, pool1_idx = self.maxpool1(pool1)
        out   = self.conv2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        layers['pool2'] = pool2
        layers['pool1_idx']  = pool1_idx
        layers['pool1_size'] = pool1.size()

        if self.level < 3: return layers

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
        
        if self.level < 4: return layers

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
        
        if self.level < 5: return layers

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

    def forward_multiple(self,x):
        out   = self.conv1_1(x)
        out   = self.relu1_1(out)
        out   = self.conv1_2(out)
        pool1 = self.relu1_2(out)
        
        layers = {'pool1': pool1}
    
        if self.level < 2: return layers

        out, pool1_idx = self.maxpool1(pool1)
        out   = self.conv2_1(out)
        out   = self.relu2_1(out)
        out   = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        layers['pool2'] = pool2

        if self.level < 3: return layers

        out, pool2_idx = self.maxpool2(pool2)

        out   = self.conv3_1(out)
        out   = self.relu3_1(out)
        out   = self.conv3_2(out)
        out   = self.relu3_2(out)
        out   = self.conv3_3(out)
        pool3 = self.relu3_3(out)
        
        layers['pool3'] = pool3

        if self.level < 4: return layers

        out, pool3_idx = self.maxpool3(pool3)

        out   = self.conv4_1(out)
        out   = self.relu4_1(out)
        out   = self.conv4_2(out)
        out   = self.relu4_2(out)
        out   = self.conv4_3(out)
        pool4 = self.relu4_3(out)

        layers['pool4'] = pool4

        if self.level < 5: layers

        out, pool4_idx = self.maxpool3(pool4)

        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)

        layers['z'] = out

        return layers

class vgg16_decoder(nn.Module):
    def __init__(self, level, prob):
        super(vgg16_decoder, self).__init__()
        self.level = level
        # [14x14]
        if level > 4:
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
            self.conv4_1 = nn.Conv2d(512,256,3,1,1)
            self.bn4_1   = nn.BatchNorm2d(256)
            self.drop4_1 = nn.Dropout(p=prob, inplace=False)
            self.relu4_1 = nn.ReLU(inplace=True)

        # [28x28]
        if level > 3:
            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv3 = nn.ConvTranspose2d(in_channels =256, 
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
            self.conv3_1 = nn.Conv2d(256,128,3,1,1)
            self.bn3_1   = nn.BatchNorm2d(128)
            self.drop3_1 = nn.Dropout(p=prob, inplace=False)
            self.relu3_1 = nn.ReLU(inplace=True)

        # [56x56]
        if level > 2:
            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv2 = nn.ConvTranspose2d(in_channels =128, 
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
            self.conv2_1 = nn.Conv2d(128,64,3,1,1)
            self.bn2_1   = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU(inplace=True)

        # [112x112]
        if level > 1:
            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.unconv1 = nn.ConvTranspose2d(in_channels =64, 
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
            #self.bn1_1   = nn.BatchNorm2d(3)
            #self.relu1_1 = nn.ReLU(inplace=True)

        # [224x224]
        if level > 0:
            self.tanh0 = nn.Tanh()

    def forward(self, layers):

        out = layers['z']
        

        if self.level > 4:
            pool4_size = layers['pool4_size']
            pool4_idx  = layers['pool4_idx']
            pool4      = layers['pool4']
            
            out = self.unpool4(out, pool4_idx, output_size=pool4_size)  # [None, 512,28,28]
            out   = self.bn4_cat1(out)
            pool4 = self.bn4_cat2(pool4)
            out = torch.cat((out,pool4), dim=1)                         # [None,1024,28,28]
            
            out = self.conv4_3(out)
            #out = self.bn4_3(out)
            #out = self.drop4_3(out)
            out = self.relu4_3(out)

            out = self.conv4_2(out)
            #out = self.bn4_2(out)
            #out = self.drop4_2(out)
            out = self.relu4_2(out)
            
            out = self.conv4_1(out)
            #out = self.bn4_1(out)
            #out = self.drop4_1(out)
            out = self.relu4_1(out)

        if self.level > 3:
            pool3_size = layers['pool3_size']
            pool3_idx  = layers['pool3_idx']
            pool3      = layers['pool3']

            out = self.unpool3(out, pool3_idx, output_size=pool3_size)
            out   = self.bn3_cat1(out)
            pool3 = self.bn3_cat2(pool3)
            out = torch.cat((out,pool3), dim=1)

            out = self.conv3_3(out)
            #out = self.bn3_3(out)
            #out = self.drop3_3(out)
            out = self.relu3_3(out)

            out = self.conv3_2(out)
            #out = self.bn3_2(out)
            #out = self.drop3_2(out)
            out = self.relu3_2(out)
            
            out = self.conv3_1(out)
            #out = self.bn3_1(out)
            #out = self.drop3_1(out)
            out = self.relu3_1(out)

        #print(out.shape)
        if self.level > 2:
            pool2_size = layers['pool2_size']
            pool2_idx  = layers['pool2_idx']
            pool2      = layers['pool2']

            out = self.unpool2(out, pool2_idx, output_size=pool2_size)
            out   = self.bn2_cat1(out)
            pool2 = self.bn2_cat2(pool2)
            out = torch.cat((out,pool2), dim=1)

            out = self.conv2_2(out)
            #out = self.bn2_2(out)
            out = self.relu2_2(out)

            out = self.conv2_1(out)
            #out = self.bn2_1(out)
            out = self.relu2_1(out)

        #print(out.shape)
        if self.level > 1:
            pool1_size = layers['pool1_size']
            pool1_idx  = layers['pool1_idx']
            pool1      = layers['pool1']

            out = self.unpool1(out, pool1_idx, output_size=pool1_size)
            out   = self.bn1_cat1(out)
            pool1 = self.bn1_cat2(pool1)

            out = torch.cat((out,pool1), dim=1)

            out = self.conv1_2(out)
            #out = self.bn1_2(out)
            out = self.relu1_2(out)

            out = self.conv1_1(out)
            #out = self.bn1_1(out)
            #out = self.relu1_1(out)

        #print(out.shape)
        if self.level > 0:
            out = self.tanh0(out)

        return out

    def forward_deconv(self, layers):
        out = layers['z']
        if self.level > 4:
            pool4 = layers['pool4']
            out   = self.unconv4(out, output_size=pool4.size())
            out   = self.bn4_cat1(out)
            pool4 = self.bn4_cat2(pool4)
            out   = torch.cat((out,pool4), dim=1)

            out = self.conv4_3(out)
            out = self.bn4_3(out)
            #out = self.drop4_3(out)
            out = self.relu4_3(out)
            out = self.conv4_2(out)
            out = self.bn4_2(out)
            #out = self.drop4_2(out)
            out = self.relu4_2(out)
            out = self.conv4_1(out)
            out = self.bn4_1(out)
            #out = self.drop4_1(out)
            out = self.relu4_1(out)

        if self.level > 3:
            pool3 = layers['pool3']
            out   = self.unconv3(out, output_size=pool3.size())
            out   = self.bn3_cat1(out)
            pool3 = self.bn3_cat2(pool3)
            out   = torch.cat((out,pool3), dim=1)

            out = self.conv3_3(out)
            out = self.bn3_3(out)
            #out = self.drop3_3(out)
            out = self.relu3_3(out)
            out = self.conv3_2(out)
            out = self.bn3_2(out)
            #out = self.drop3_2(out)
            out = self.relu3_2(out)
            out = self.conv3_1(out)
            out = self.bn3_1(out)
            #out = self.drop3_1(out)
            out = self.relu3_1(out)

        if self.level > 2:
            pool2 = layers['pool2']
            out   = self.unconv2(out, output_size=pool2.size())
            out   = self.bn2_cat1(out)
            pool2 = self.bn2_cat2(pool2)
            out   = torch.cat((out,pool2), dim=1)

            out = self.conv2_2(out)
            out = self.bn2_2(out)
            out = self.relu2_2(out)
            out = self.conv2_1(out)
            out = self.bn2_1(out)
            out = self.relu2_1(out)

        if self.level > 1:
            pool1 = layers['pool1']
            out   = self.unconv1(out, output_size=pool1.size())
            out   = self.bn1_cat1(out)
            pool1 = self.bn1_cat2(pool1)
            out   = torch.cat((out,pool1), dim=1)

            out = self.conv1_2(out)
            out = self.bn1_2(out)
            out = self.relu1_2(out)
            out = self.conv1_1(out)
            #out = self.relu1_1(out)

        if self.level > 0:
            out = self.tanh0(out)

        return out