import utils.dataloaders as dataloaders
import numpy as np
from utils.arg_extractor import get_args
from utils.new_experiment_builder import ExperimentBuilder
from utils.models import ConvLSTMModel
import torch
from torch.utils.data import DataLoader
args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  

torch.manual_seed(seed=args.seed)
args.toy = False
train_dataset = dataloaders.MilanDataLoader(_set = 'train',toy = args.toy)
valid_dataset = dataloaders.MilanDataLoader(_set = 'valid',toy = args.toy)
test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = args.toy)



train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last = True)


valid_data = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last = True)


test_data = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last = True)

seq_start = 12
seq_length = 18

model = ConvLSTMModel(input_size = args.image_height, seq_start = args.seq_start, seq_length = args.seq_length, batch_size = args.batch_size,use_gpu=args.use_gpu)
experiment = ExperimentBuilder(network_model=model,seq_start = seq_start,seq_length = seq_length,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    lr =args.learning_rate, weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    device=device,
                                    train_data=train_data, val_data=valid_data,
                                    test_data=test_data)  # build an experiment object
experiment_metrics, test_metrics = experiment.run_experiment()