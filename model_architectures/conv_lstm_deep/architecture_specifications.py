from model_architectures.conv_lstm_deep.DeepConvLstm import ConvLSTM
from collections import OrderedDict

def encoder_architecture(batch_size, device, seq_input): 
    
    return [
        [ #in_channels, out_channels, kernel_size, stride, padding
            OrderedDict({'conv1_leaky_1': [1, 8, 4, 2, 1]}),
            OrderedDict({'conv2_leaky_1': [64, 192, 4, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
        ],

        [
            ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 50, 50),
                    kernel_size=3, stride=1, padding=1,device=device,seq_len=seq_input),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 25, 25),
                    kernel_size=3, stride=1, padding=1,device=device,seq_len=seq_input),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 13, 13),
                    kernel_size=3, stride=1, padding=1,device=device,seq_len=seq_input),
        ]
    ]
###### Define decoder #####
def forecaster_architecture(batch_size, device, seq_output):  
    return [
        [
            OrderedDict({'deconv1_leaky_1': [192, 192, 3, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
            OrderedDict({
                'deconv3_leaky_1': [64, 8, 4, 2, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            }),
        ],

        [
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 13, 13),
                    kernel_size=3, stride=1, padding=1,device=device,seq_len=seq_output),
            ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 25, 25),
                    kernel_size=3, stride=1, padding=1,device=device,seq_len=seq_output),
            ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 50, 50),
                    kernel_size=3, stride=1, padding=1,device=device,seq_len=seq_output),
        ]
    ]
