import torch
import numpy as np
from utils.model_loader import create_model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils.dataloaders as dataloaders

class args_class(object):
    def __init__(self,batch_size,seq_input,seq_output):
        self.batch_size = batch_size
        self.seq_start  = seq_input
        self.seq_output = seq_output 
        self.seq_length = seq_input + seq_output
        self.image_height = 100
        self.use_gpu = False

def load_pytorch_model_to_cpu(model,PARAMS_PATH):
    network = torch.load(PARAMS_PATH,map_location='cpu')['network']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key,value in network.items():
        name = key[6:]
        new_state_dict[name] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

RESULTS_PATH = '/home/s1818503/dissertation/Mobile-Data-Forecasting-With-Spatio-Temporal-Networks/experiments_results/'
TEST_SET_PATH = '/home/s1818503/dissertation/Mobile-Data-Forecasting-With-Spatio-Temporal-Networks/data_in12_out6/milan_processed_test.npz'
# RESULTS_PATH = "/home/nick/Desktop/experiments_results/"
# TEST_SET_PATH = "/home/nick/Desktop/experiments_results/milan_processed_test.npz"
# RESULTS_PATH  = '/afs/inf.ed.ac.uk/user/s18/s1818503/Desktop/experiments_results/'
# TEST_SET_PATH = '/afs/inf.ed.ac.uk/user/s18/s1818503/Desktop/experiments_results/milan_processed_test.npz'
RESULT_FOLDERS = {
#     'no_scaling' : "conv_lstm_results_before_scaling_i=5_o=15/",
#     'standard'   : "conv_lstm_with_data_standardisation_i=5_o=15/"
    # 'deepconvlstm' :  'hzzone_conv_lstm/' ,
    # 'shallowconvlstm' :  'cxiixi_conv_lstm/'
    'deepconvlstm' :  '' ,
    'shallowconvlstm' :  ''
}

experiment_name = 'conv_lstm_lr_-3'
device='cuda'
args  =  args_class(10 ,12, 6)
model_name = 'shallowconvlstm'

PARAMS_PATH = RESULTS_PATH + RESULT_FOLDERS[model_name] + experiment_name + '/saved_models/train_model_latest'
model = create_model(model_name,args,device)
model = load_pytorch_model_to_cpu(model,PARAMS_PATH)

torch.manual_seed(seed=1)
test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = False, DATA_DIR=TEST_SET_PATH)
test_data = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last = True)
_ , y = test_dataset.__getitem__(1)
mse_frame_timestep = torch.zeros(y.shape[0])

for x,y in test_data:
    out = model.forward(x)
    se_batch = torch.sum((out.squeeze() - y)**2,(2,3))
    mse_frame_timestep += torch.mean(se_batch,0) 

mse_frame_timestep = mse_frame_timestep / len(test_data)
numpy.savetxt(RESULTS_PATH + experiment_name + '/mse_frame_timestep.csv', mse_frame_timestep, delimiter=",")
fig = plt.figure()
plt.plot(mse_frame_timestep.detach().numpy(),'-o')
plt.title('MSE/frame ' + model_name)
plt.xlabel('timestep')
plt.ylabel('MSE')
plt.savefig(RESULTS_PATH + 'figures/' + experiment_name + '.pdf')