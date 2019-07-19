import utils.dataloaders as dataloaders
import numpy as np
from utils.arg_extractor import get_args
from utils.experiment_builder import ExperimentBuilder
from utils.model_loader import create_model
import torch
from torch.utils.data import DataLoader
args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  

torch.manual_seed(seed=args.seed)
args.toy = False
#load dataset
train_dataset = dataloaders.MilanDataLoader(_set = 'train',toy = args.toy,cropped = args.cropped)
valid_dataset = dataloaders.MilanDataLoader(_set = 'valid',toy = args.toy,cropped = args.cropped)
test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = args.toy,cropped = args.cropped)

train_data = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
valid_data = DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
test_data = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
#load model
model = create_model(args.model,args,device)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_algorithm, mode='min', factor=0.1, patience=7)
experiment = ExperimentBuilder(network_model=model,seq_start = args.seq_start,seq_length = args.seq_length,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    lr =args.learning_rate, weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    device=device,grad_clip=args.grad_clip,
                                    train_data=train_data, val_data=valid_data,
                                    test_data=test_data)  # build an experiment object
experiment_metrics, test_metrics = experiment.run_experiment()