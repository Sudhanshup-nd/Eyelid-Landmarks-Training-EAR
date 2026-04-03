import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetDec(nn.Module):
    def __init__(self, inchannels, skipconnchannels, outchannels):
        super(UnetDec, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(skipconnchannels + inchannels//2, outchannels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannels)

    def forward(self, x, skipcon):
     #   print('x in =', x.shape)
        x = self.upconv1(x)
     #   print('x upconv out =', x.shape)
        x = torch.cat([x, skipcon], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class UnetEnc(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(UnetEnc, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) #F.relu(self.bn1(self.conv1(x)))
        x_mp = self.pool(x)
        return x, x_mp

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        num_seg_classes = 4
        self.enc64 = UnetEnc(1, 16)   # => 16
        self.enc32 = UnetEnc(16, 16)  # => 16
        self.enc16 = UnetEnc(16, 24)  # => 24
        self.enc8 = UnetEnc(24, 32)   # => 32

        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.dec8  = UnetDec(32, 32, 32)
        self.dec16 = UnetDec(32, 24, 24)
        self.dec32 = UnetDec(24, 16, 16)
        self.dec64 = UnetDec(16, 16, 16)
        self.seg1 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.seg2 = nn.Conv2d(8, num_seg_classes, 1, stride=1)

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('---unet segmentor---')
        print('NUM PARAMS =', self.num_params)
        print()


    def forward(self, x):
        x64, x = self.enc64(x)
        x32, x = self.enc32(x)
        x16, x = self.enc16(x)
        x8, x = self.enc8(x)
        x = self.conv(x)
        print('dec8')
        x = self.dec8(x, x8)
        print('dec16')
        x = self.dec16(x, x16)
        print('dec32')
        x = self.dec32(x, x32)
        print('dec64')
        x = self.dec64(x, x64)
        x = self.seg2(self.seg1(x))
        return x

    def set_train(self):
        pass

    def set_eval(self):
        pass


class UNetBackbone(nn.Module):
    """
    UNet backbone for feature extraction. Used for landmarks heads.
    - Accepts 1-channel input like original UNet.
    - forward returns a global pooled feature vector.
    - freeze_backbone() sets requires_grad=False for backbone params.
    """
    def __init__(self):
        super().__init__()
        self.enc64 = UnetEnc(1, 16)
        self.enc32 = UnetEnc(16, 16)
        self.enc16 = UnetEnc(16, 24)
        self.enc8 = UnetEnc(24, 32)
        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.out_channels = 32

    def forward(self, x):
        _, x = self.enc64(x)
        _, x = self.enc32(x)
        _, x = self.enc16(x)
        _, x = self.enc8(x)
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False


class UnetUpSampleDec(nn.Module):
    def __init__(self, inchannels, skipconnchannels, outchannels):
        super(UnetUpSampleDec, self).__init__()
        self.upsamp1 = nn.Upsample(scale_factor=2)
        self.conv0 = nn.Conv2d(inchannels, inchannels//2, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(skipconnchannels + inchannels//2, outchannels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outchannels)

    def forward(self, x, skipcon):
        x = self.conv0(self.upsamp1(x))
        x = torch.cat([x, skipcon], dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x))) #F.relu(self.bn1(self.conv1(x)))
        return x


# ## OLD MODEL - trained by Dhruv
# class UnetUpSample(nn.Module):
#     def __init__(self):
#         super(UnetUpSample, self).__init__()

#         num_seg_classes = 4 
#         self.enc64 = UnetEnc(1, 16)   # => 16 
#         self.enc32 = UnetEnc(16, 16)  # => 16
#         self.enc16 = UnetEnc(16, 24)  # => 24
#         self.enc8 = UnetEnc(24, 32)   # => 32

#         self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)

#         self.dec8  = UnetUpSampleDec(32, 32, 32)
#         self.dec16 = UnetUpSampleDec(32, 24, 24)
#         self.dec32 = UnetUpSampleDec(24, 16, 16)
#         self.dec64 = UnetUpSampleDec(16, 16, 16)
#         self.seg1 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
#         self.seg2 = nn.Conv2d(8, num_seg_classes, 1, stride=1)

#         self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print('---unet segmentor with upsample layers---')
#         print('NUM PARAMS =', self.num_params)
#         print()


#     def forward(self, x):
#         x64, x = self.enc64(x)
#         x32, x = self.enc32(x)
#         x16, x = self.enc16(x)
#         x8, x = self.enc8(x)
#         x = self.conv(x)

#         x = self.dec8(x, x8)
#         x = self.dec16(x, x16)
#         x = self.dec32(x, x32)
#         x = self.dec64(x, x64)
#         x = self.seg2(self.seg1(x))
#         return x

#     def set_train(self):
#         pass

#     def set_eval(self):
#         pass
   
# NEW MODEL - for training
class UnetUpSample(nn.Module):  
    def __init__(self):  
        super(UnetUpSample, self).__init__()  
  
        num_seg_classes = 5  # Change number of output channels to 5  
        self.enc64 = UnetEnc(1, 16)   # => 16  
        self.enc32 = UnetEnc(16, 16)  # => 16  
        self.enc16 = UnetEnc(16, 24)  # => 24  
        self.enc8 = UnetEnc(24, 32)   # => 32  
  
        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)  
  
        self.dec8  = UnetUpSampleDec(32, 32, 32)  
        self.dec16 = UnetUpSampleDec(32, 24, 24)  
        self.dec32 = UnetUpSampleDec(24, 16, 16)  
        self.dec64 = UnetUpSampleDec(16, 16, 16)  
        self.seg1 = nn.Conv2d(16, 8, 3, stride=1, padding=1)  
        self.seg2 = nn.Conv2d(8, num_seg_classes, 1, stride=1)  
  
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  
        print('---unet segmentor with upsample layers---')  
        print('NUM PARAMS =', self.num_params)  
        print()  
  
    def forward(self, x):  
        x64, x = self.enc64(x)  
        x32, x = self.enc32(x)  
        x16, x = self.enc16(x)  
        x8, x = self.enc8(x)  
        x = self.conv(x)  
  
        x = self.dec8(x, x8)  
        x = self.dec16(x, x16)  
        x = self.dec32(x, x32)  
        x = self.dec64(x, x64)  
        x = self.seg2(self.seg1(x))  
  
        # create a dictionary to store the output  
        output = {"out": x}  
  
        # return the output in the desired manner  
        return output["out"][:,:4,:,:], output["out"][:,4,:,:]  
  
    def set_train(self):  
        pass  
  
    def set_eval(self):  
        pass  


# NEW MODEL - for testing as sigmoid applied on output2 - used in onnx conversion and also using for retraining the eye segmentation model with blink suppression
class UnetUpSample_modified(nn.Module):  
    def __init__(self):  
        super(UnetUpSample_modified, self).__init__()  
  
        num_seg_classes = 5  # Change number of output channels to 5  
        self.enc64 = UnetEnc(1, 16)   # => 16  
        self.enc32 = UnetEnc(16, 16)  # => 16  
        self.enc16 = UnetEnc(16, 24)  # => 24  
        self.enc8 = UnetEnc(24, 32)   # => 32  
  
        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)  
   
        self.dec8  = UnetUpSampleDec(32, 32, 32)  
        self.dec16 = UnetUpSampleDec(32, 24, 24)  
        self.dec32 = UnetUpSampleDec(24, 16, 16)  
        self.dec64 = UnetUpSampleDec(16, 16, 16)  
        self.seg1 = nn.Conv2d(16, 8, 3, stride=1, padding=1)  
        self.seg2 = nn.Conv2d(8, num_seg_classes, 1, stride=1)  
  
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  
        print('---unet segmentor with upsample layers---')  
        print('NUM PARAMS =', self.num_params)  
        print()  
  
    def forward(self, x):  
        x64, x = self.enc64(x)  
        x32, x = self.enc32(x)  
        x16, x = self.enc16(x)  
        x8, x = self.enc8(x)  
        x = self.conv(x)  
  
        x = self.dec8(x, x8)   
        x = self.dec16(x, x16)  
        x = self.dec32(x, x32)  
        x = self.dec64(x, x64)  
        x = self.seg2(self.seg1(x))  
  
        x_op1 = x[:,:4,:,:]
        x_softmax = F.softmax(x_op1, dim=1)  
        output1 = torch.argmax(x_softmax, dim=1)  
          
        x_op2 = x[:,-1:,:,:]   ## SHarf

        output2 = F.sigmoid(x_op2)
        
        return output1, output2
        
  
    def set_train(self):  
        pass  
  
    def set_eval(self):  
        pass  

