import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_architectures.conv_lstm_shallow.ConvLstmCell import ConvLSTMCell
from torch.autograd import Variable
from torch.nn import functional as func
""" modified code from  https://github.com/cxxixi/Precipitation-Nowcasting """

class ConvLSTMModel(nn.Module):

    def __init__(self,input_size,seq_start,seq_length, batch_size,use_gpu):
        super(ConvLSTMModel,self).__init__()
        self.input_dim = 'BSHW' # BATCH_SIZE, SEQUENCE_START, HEIGHT, WIDTH
        self.seq_start = seq_start
        self.seq_length = seq_length
        self.batch_size = batch_size
        #declare some parameters that might be used 
        self.conv_pad = 0
        self.conv_kernel_size = 3
        self.conv_stride = 1
        self.pool_pad = 0
        self.pool_kernel_size = 3
        self.pool_stride = 3
        self.hidden_size = 64
        self.size = int((input_size+2*self.conv_pad-(self.conv_kernel_size-1)-1)/self.conv_stride+1)
        self.size1 = int((self.size+2*self.pool_pad-(self.pool_kernel_size-1)-1)/self.pool_stride+1)
        #define layers
        self.use_gpu = use_gpu
        self.conv = nn.Conv2d(
             in_channels=1,
             out_channels=8,
             kernel_size=3,
             stride=1,
             padding=0)
        self.pool = nn.MaxPool2d(
                     kernel_size=3
                     )
        self.convlstm1 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=8, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.convlstm2 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=self.hidden_size, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.deconv = nn.ConvTranspose2d(
                        in_channels=self.hidden_size , 
                        out_channels=1, 
                        kernel_size=6,
                        stride=3,
                        padding=0, 
                        output_padding=1, 
                        )
        self.relu = func.relu


    def forward(self,X):
        X_chunked = torch.chunk(X,self.seq_start,dim=1)
        X = None
        output = [None]*self.seq_length
        state_size = [self.batch_size, self.hidden_size]+[self.size1,self.size1]
        if self.use_gpu == False:
            hidden1 = Variable(torch.zeros(state_size))
            cell1 = Variable(torch.zeros(state_size))
            hidden2 = Variable(torch.zeros(state_size))
            cell2 = Variable(torch.zeros(state_size))
        else :
            hidden1 = Variable(torch.zeros(state_size)).cuda()
            cell1 = Variable(torch.zeros(state_size)).cuda()
            hidden2 = Variable(torch.zeros(state_size)).cuda()
            cell2 = Variable(torch.zeros(state_size)).cuda()

        for i in range(self.seq_start):

            output[i] = self.conv(X_chunked[i])     
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            # output[i] = self.relu(output[i])
        
        for i in range(self.seq_start,self.seq_length):                                                 
            output[i] = self.conv(output[i-1])    
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            # output[i] = self.relu(output[i])
            
        out = torch.stack(output[self.seq_start:]).squeeze()
        return out.permute(1,0,2,3)