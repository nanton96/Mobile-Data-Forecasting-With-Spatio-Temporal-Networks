# import sys
# sys.path.insert(0, '../')
# import torch
# from collections import OrderedDict
# from utils.new_models import EF,Encoder,Forecaster,ConvLSTM
# import utils.dataloaders as dataloaders
# batch_size = 10
# encoder_architecture = [
#     [   OrderedDict({'conv1_leaky_1': [1, 8, 4, 2, 1]}),
#         OrderedDict({'conv2_leaky_1': [64, 192, 4, 2, 1]}),
#         OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
#     ],

#     [
#         ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 50, 50),
#                  kernel_size=3, stride=1, padding=1),
#         ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 25, 25),
#                  kernel_size=3, stride=1, padding=1),
#         ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 13, 13),
#                  kernel_size=3, stride=1, padding=1),
#     ]
# ]
# encoder = Encoder(encoder_architecture[0],encoder_architecture[1])

# encoder


import utils.dataloaders as dataloaders
import numpy as np
from utils.arg_extractor import get_args
from utils.new_experiment_builder import ExperimentBuilder
from utils.new_models import EF,Encoder,Forecaster,ConvLSTM
import torch
from torch.utils.data import DataLoader
args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  
from collections import OrderedDict

torch.manual_seed(seed=args.seed)
args.toy = False
batch_size = args.batch_size

train_dataset = dataloaders.MilanDataLoader(_set = 'train',toy = args.toy,create_channel_axis=False)
valid_dataset = dataloaders.MilanDataLoader(_set = 'valid',toy = args.toy,create_channel_axis=True)
test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = args.toy,create_channel_axis=True)

train_data = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
valid_data = DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
test_data = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)


for i,(x,y) in enumerate(train_data):
    print(i)