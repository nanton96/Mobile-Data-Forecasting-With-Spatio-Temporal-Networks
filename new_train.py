import utils.dataloaders as dataloaders
import numpy as np
from utils.arg_extractor import get_args
from utils.experiment_builder import ExperimentBuilder

from model_architectures.conv_lstm_deep.DeepConvLstm import EF,Encoder,Forecaster,ConvLSTM
from model_architectures.conv_lstm_deep.architecture_specifications import encoder_architecture, forecaster_architecture

import torch
from torch.utils.data import DataLoader
args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  
from collections import OrderedDict

torch.manual_seed(seed=args.seed)
args.toy = False
batch_size = args.batch_size

train_dataset = dataloaders.MilanDataLoader(_set = 'train',toy = args.toy,create_channel_axis=False)
valid_dataset = dataloaders.MilanDataLoader(_set = 'valid',toy = args.toy,create_channel_axis=False)
test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = args.toy,create_channel_axis=False)

train_data = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
valid_data = DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
test_data = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)


seq_input = 12
seq_output = 6
seq_length = 18

###### Define encoder #####
enc_arch = encoder_architecture(batch_size, device, seq_input)
encoder = Encoder(enc_arch[0],enc_arch[1]).to(device)
###### Define decoder #####
fore_arch = forecaster_architecture(batch_size, device, seq_output)
forecaster=Forecaster(fore_arch[0],fore_arch[1],seq_output).to(device)

model = EF(encoder,forecaster)

experiment = ExperimentBuilder(network_model=model,seq_start = seq_input,seq_length = seq_length,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    lr =args.learning_rate, weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    device=device,
                                    train_data=train_data, val_data=valid_data,
                                    test_data=test_data)  # build an experiment object

experiment_metrics, test_metrics = experiment.run_experiment()