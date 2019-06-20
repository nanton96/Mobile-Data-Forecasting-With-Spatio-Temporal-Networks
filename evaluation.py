import utils.dataloaders as dataloaders
import numpy as np
# from utils.arg_extractor import get_args
from utils.new_experiment_builder import ExperimentBuilder
from utils.new_models import EF,Encoder,Forecaster,ConvLSTM
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

# args, device = get_args()  # get arguments from command line

device = 'cpu'
rng = np.random.RandomState(12345)  
from collections import OrderedDict

batch_size = 10

test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = False,create_channel_axis=True, DATA_DIR = '/home/nick/Desktop/experiments_results/milan_processed_test.npz')
test_data = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,drop_last = True)

seq_input = 12
seq_output = 6
seq_length = 18
###### Define encoder #####
encoder_architecture = [
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
encoder = Encoder(encoder_architecture[0],encoder_architecture[1]).to(device)
###### Define decoder #####
a = OrderedDict()
a['deconv3_leaky_1'] = [64, 8, 4, 2, 1]
a['conv3_leaky_2'] = [8, 8, 3, 1, 1]
a['conv3_3']      = [8, 1, 1, 1, 0]

forecaster_architecture = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 3, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
        a,
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
forecaster=Forecaster(forecaster_architecture[0],forecaster_architecture[1],seq_output).to(device)

model_ef = EF(encoder,forecaster)