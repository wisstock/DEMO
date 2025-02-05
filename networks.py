# Convolutional neural networks for semantic segmentation of grayscales images

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import Sequential
from torch.nn import Conv2d, Dropout2d, MaxPool2d, ReLU, UpsamplingNearest2d



# Classic UNet 
class UNetNaive(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last baseline block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512


        # Baseline
        # input: 32x32x512
        self.e51 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(32, n_class, kernel_size=1)


    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True))
        
        self.encoder1 = conv_block(in_channels, 32)
        self.encoder2 = conv_block(32, 64)
        self.encoder3 = conv_block(64, 128)
        self.encoder4 = conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        self.baseline1 = conv_block(256, 512)
        self.baseline2 = conv_block(512, 512)
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Middle path
        base1 = self.baseline1(self.pool(enc4))
        base2 = self.baseline2(base1)
        
        # Decoder path
        dec4 = torch.cat((self.upconv4(base2), enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = torch.cat((self.upconv3(dec4), enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = torch.cat((self.upconv2(dec3), enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = torch.cat((self.upconv1(dec2), enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.out_conv(dec1)
        return out


# Compressed UNet
class UNetMiniNaive(nn.Module):

    def __init__(self, num_classes):
        super(UNetMiniNaive, self).__init__()

        # Use padding 1 to mimic `padding='same'` in keras,
        # use this visualization tool https://ezyang.github.io/convolution-visualizer/index.html
        self.block1 = Sequential(
            Conv2d(1, 32, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool1 = MaxPool2d((2, 2))

        self.block2 = Sequential(
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU(),
        )
        self.pool2 = MaxPool2d((2, 2))

        self.block3 = Sequential(
            Conv2d(64, 128, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU()
        )

        self.up1 = UpsamplingNearest2d(scale_factor=2)
        self.block4 = Sequential(
            Conv2d(192, 64, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(64, 64, kernel_size=3, padding=1),
            ReLU()
        )

        self.up2 = UpsamplingNearest2d(scale_factor=2)
        self.block5 = Sequential(
            Conv2d(96, 32, kernel_size=3, padding=1),
            ReLU(),
            Dropout2d(0.2),
            Conv2d(32, 32, kernel_size=3, padding=1),
            ReLU()
        )

        self.conv2d = Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)
        out_pool2 = self.pool1(out2)

        out3 = self.block3(out_pool2)

        out_up1 = self.up1(out3)
        # return out_up1
        out4 = torch.cat((out_up1, out2), dim=1)
        out4 = self.block4(out4)

        out_up2 = self.up2(out4)
        out5 = torch.cat((out_up2, out1), dim=1)
        out5 = self.block5(out5)

        out = self.conv2d(out5)

        return out
    

class UNetMini(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetMini, self).__init__()
        
        def conv_block(in_ch, out_ch, dropout_prob=0.5):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(p=dropout_prob),
                                 nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(p=dropout_prob))
        
        self.encoder1 = conv_block(in_channels, 32)
        self.encoder2 = conv_block(32, 64)
        # self.encoder3 = conv_block(64, 128)
        
        self.pool = nn.MaxPool2d(2)
        
        self.baseline1 = conv_block(64, 128)
        self.baseline2 = conv_block(128, 128)
        # self.baseline3 = conv_block(64, 64)
        # self.baseline4 = conv_block(128, 128)

        # self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # self.decoder3 = conv_block(256, 128)        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        # enc3 = self.encoder3(self.pool(enc2))
        
        # Middle path
        base1 = self.baseline1(self.pool(enc2))
        base2 = self.baseline2(base1)
        # base3 = self.baseline2(base2)
        # base4 = self.baseline2(base3)
        
        # Decoder path
        # dec3 = torch.cat((self.upconv3(base2), enc3), dim=1)
        # dec3 = self.decoder3(dec3)

        dec2 = torch.cat((self.upconv2(base2), enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = torch.cat((self.upconv1(dec2), enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.out_conv(dec1)
        return out