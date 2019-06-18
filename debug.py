import sys
sys.path.insert(0, '../')
import torch
from collections import OrderedDict
from utils.new_models import EF,Encoder,Forecaster,ConvLSTM

batch_size = 10
encoder_architecture = [
    [   OrderedDict({'conv1_leaky_1': [1, 8, 4, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 4, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 50, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 25, 25),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 13, 13),
                 kernel_size=3, stride=1, padding=1),
    ]
]
encoder = Encoder(encoder_architecture[0],encoder_architecture[1])

encoder