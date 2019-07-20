

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    
    def __init__(self):

        super(CNN3D,self).__init__ ()
        self.conv1   = nn.Conv3d(in_channels=1,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.conv2   = nn.Conv3d(in_channels=64,out_channels=128,kernel_size=3,stride=(1,2,2),padding=1)
        self.conv3   = nn.Conv3d(in_channels=128,out_channels=128,kernel_size=3,stride=(2,2,2),padding=1)
        self.deconv1 = nn.ConvTranspose3d(in_channels=128,out_channels=128,kernel_size=3,stride=(2,2,2),padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=(5,4,4),stride=(1,2,2),padding=1,output_padding=0)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64,out_channels=1,kernel_size=5,stride=1,padding=2,output_padding=0)

    def forward(self,x): 
        
        x.unsqueeze(1) # BSHW to BCSHW

        out = self.conv1(x)  
        out = self.conv2(out)  
        out = self.conv3(out)  
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)

        return out.squeeze() # BCSHW to BSHW