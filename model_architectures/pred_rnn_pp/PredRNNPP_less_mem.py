# based on the tensorflow implementation by 'yunbo':
# https://github.com/Yunbo426/predrnn-pp/blob/master/nets/predrnn_pp.py

import torch
import torch.nn as nn
from model_architectures.pred_rnn_pp.CausalLSTMCell_less_mem import CausalLSTMCell as clstm
from model_architectures.pred_rnn_pp.GradientHighWayUnit import GHU

class PredRNNPP(nn.Module):

    def __init__(self,input_shape,seq_input,seq_output,batch_size,num_hidden,device,use_GHU = False):
        super(PredRNNPP,self).__init__()

        self.seq_input = seq_input
        self.seq_output = seq_output
        self.seq_length = seq_input + seq_output
        self.device = device
        self.batch_size = batch_size
        self.input_shape = input_shape #this is the dimensionality of the frame
        self.num_hidden = num_hidden
        self.num_layers = len(num_hidden)

        self.lstm = nn.ModuleList()
        self.output_channels = 1
        self.conv = nn.Conv2d(in_channels=1,
                           out_channels=8, 
                           kernel_size=3,
                           stride=1,
                           padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3)

        self.compressed_shape = [batch_size,8,32,32]
        self.use_GHU = use_GHU
        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = self.num_hidden[self.num_layers-1]
                input_channels = 8
            else:
                num_hidden_in = self.num_hidden[i-1]
                input_channels = self.num_hidden[i-1]

            new_cell = clstm(input_channels,'lstm_'+str(i+1),3,num_hidden_in,self.num_hidden[i],self.compressed_shape,self.device)
            self.lstm.append(new_cell)
        
        if self.use_GHU:
            self.ghu = GHU(filter_size=3, num_features = num_hidden[1],input_channels=num_hidden[0])

        self.deconv = nn.ConvTranspose2d(
            in_channels= num_hidden[len(num_hidden)-1] , 
            out_channels=1, 
            kernel_size=6,
            stride=3,
            padding=0, 
            output_padding=1 
        )

    def forward(self,x):

        cell = []
        hidden = []
        mem = None
        for i in range(self.num_layers):
            cell.append(None)
            hidden.append(None)
        output = []
        x_gen = None
        # x has shape B S H W
        for t in range(self.seq_length):

            if t < self.seq_input:
                inputs = x[:,t,:,:].unsqueeze(1)
            else:
                inputs = x_gen

            inputs = self.conv(inputs)  #to 98x98
            inputs = self.pool(inputs)  #to 32x32
            # Causal LSTMs do not change dimensionality
            hidden[0], cell[0], mem = self.lstm[0].forward(inputs, hidden[0],cell[0], mem)
            
            if self.use_GHU:
                z_t = self.ghu(self.hidden[0], z_t)
            else:
                z_t = hidden[0]
            hidden[1],cell[1],mem = self.lstm[1](z_t, hidden[1], cell[1], mem)
            for i in range(2, self.num_layers):
                hidden[i], cell[i], mem = self.lstm[i](hidden[i-1], hidden[i], cell[i], mem)
            
            x_gen = self.deconv(hidden[self.num_layers-1]) #back to 100x100
            output.append(x_gen.squeeze())
            # print('t= ', t, ' memory :', torch.cuda.max_memory_allocated())

        output = torch.stack(output[self.seq_input:])
        if self.batch_size==1:
            output = output.unsqueeze(1)
        return output.permute(1,0,2,3)