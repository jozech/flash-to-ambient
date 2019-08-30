import torch.nn as nn

class vgg_encoder(nn.Module):
    def __init__(self, level):
        super(vgg_encoder, self).__init__()
        self.level = level

#        self.conv0 = nn.Conv2d( in_channels =3,
#                                out_channels=3,
#                                kernel_size =1,
#                                stride      =1,
#                                padding     =0)

        self.pad1_1 = nn.ReflectionPad2d((1,1,1,1)) # [left, right, top, bottom]

        self.conv1_1 = nn.Conv2d(3,64,3,1,0)
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
        #out = self.conv0(x)

        #print(x.shape)
        out = self.pad1_1(x)
        out = self.conv1_1(out)
        out = self.relu1_1(out)
        #print(out.shape)

        if self.level < 2: 
            return out

        out   = self.pad1_2(out)
        out   = self.conv1_2(out)
        pool1 = self.relu1_2(out)
        #print(out.shape)

        out, pool1_idx = self.maxpool1(pool1)
        #print(out.shape)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)
        #print(out.shape)

        if self.level < 3: 
            return out, pool1_idx, poo1.size()

        out   = self.pad2_2(out)
        out   = self.conv2_2(out)
        pool2 = self.relu2_2(out)
        #print(out.shape)

        out, pool2_idx = self.maxpool2(pool2)
        #print(out.shape)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)
        #print(out.shape)

        if self.level < 4:
            return out, pool1_idx, pool1.size(), pool2_idx, pool2.size()

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        #print(out.shape)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        #print(out.shape)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        #print(out.shape)

        out, pool3_idx = self.maxpool3(pool3)
        #print(out.shape)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)
        #print(out.shape)

        return out, pool1_idx, pool1.size(), pool2_idx, pool2.size(), pool3_idx, pool3.size()

    def forward_multiple(self, x):
        #out = self.conv0(x)

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

class vgg_decoder(nn.Module):
    def __init__(self, level):
        super(vgg_decoder, self).__init__()
        self.level = level

        if level > 3:
            self.pad4_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv4_1 = nn.Conv2d(512,256,3,1,0)
            self.relu4_1 = nn.ReLU(inplace=True)     #[28,28]

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            
            self.pad3_4  = nn.ReflectionPad2d((1,1,1,1))
            self.conv3_4 = nn.Conv2d(256,256,3,1,0)
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
            
            self.pad2_2  = nn.ReflectionPad2d((1,1,1,1))
            self.conv2_2 = nn.Conv2d(128,128,3,1,0)
            self.relu2_2 = nn.ReLU(inplace=True)

        if level > 1:
            self.pad2_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu2_1 = nn.ReLU(inplace=True)

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad1_2  = nn.ReflectionPad2d((1,1,1,1))
            self.conv1_2 = nn.Conv2d(64,64,3,1,0)
            self.relu1_2 = nn.ReLU(inplace=True)

        if level > 0:
            self.pad1_1  = nn.ReflectionPad2d((1,1,1,1))
            self.conv1_1 = nn.Conv2d(64,3,3,1,0)


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
        #print('\nDecoder:')
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
            #print(out.shape)

        return out
