import torch
import numpy as np
from utils.model_loader import create_model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import utils.dataloaders as dataloaders
import tqdm
import matplotlib
matplotlib.use('Agg')
import argparse


class args_class(object):
    def __init__(self,batch_size,seq_input,seq_output):
        self.batch_size = batch_size
        self.seq_start  = seq_input
        self.seq_output = seq_output 
        self.seq_length = seq_input + seq_output
        self.image_height = 100
        self.use_gpu = True

def load_pytorch_model_to_gpu(model,PARAMS_PATH):
    network = torch.load(PARAMS_PATH,map_location='cuda')['network']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key,value in network.items():
        name = key[6:] #because model dict keys are in the form model. (need to remove first 6 characters to get proper names)
        new_state_dict[name] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

### GET ARGUMENTS FROM COMMAND LINE
parser = argparse.ArgumentParser(
        description='evaluation script')
parser.add_argument('--experiment_name', type=str,default='new_conv_lstm_lr_-3_in12_out10_no_shuffle_before_split',help='name of experiment to evaluate')
parser.add_argument('--machine', type=str,default='cluster', help='name of machine where the script is run')
arguments = parser.parse_args()

if arguments.experiment_name.split('_')[0] == 'new':
    model_name = 'deepconvlstm'
elif arguments.experiment_name.split('_')[0] == 'conv':
    model_name = 'shallowconvlstm'

if arguments.machine.lower() =='cluster':
    RESULTS_PATH = '/home/s1818503/dissertation/Mobile-Data-Forecasting-With-Spatio-Temporal-Networks/experiments_results/'
    TEST_SET_PATH = '/home/s1818503/dissertation/Mobile-Data-Forecasting-With-Spatio-Temporal-Networks/data/milan_processed_test.npz' #data is with input 12 and output 10
elif arguments.machine.lower() =='personal':
    RESULTS_PATH = "/home/nick/Desktop/experiments_results/"
    TEST_SET_PATH = "/home/nick/Desktop/experiments_results/milan_processed_test.npz"
elif arguments.machine.lower() == 'dice':    
    RESULTS_PATH  = '/afs/inf.ed.ac.uk/user/s18/s1818503/Desktop/experiments_results/'
    TEST_SET_PATH = '/afs/inf.ed.ac.uk/user/s18/s1818503/Desktop/experiments_results/milan_processed_test.npz'

experiment_name = arguments.experiment_name

RESULT_FOLDERS = {
#     'no_scaling' : "conv_lstm_results_before_scaling_i=5_o=15/",
#     'standard'   : "conv_lstm_with_data_standardisation_i=5_o=15/"
    # 'deepconvlstm' :  'hzzone_conv_lstm/' ,
    # 'shallowconvlstm' :  'cxiixi_conv_lstm/'
    'deepconvlstm' :  '' ,
    'shallowconvlstm' :  ''
}

#### THIS NEEDS TO BE AN ARGUMENT
device = torch.cuda.current_device()
#### NEED TO CHANGE THIS to read meta data from experiment####
args  =  args_class(5 ,12, 10)


PARAMS_PATH = RESULTS_PATH + RESULT_FOLDERS[model_name] + experiment_name + '/saved_models/train_model_latest'
model = create_model(model_name,args,device)
model = load_pytorch_model_to_gpu(model,PARAMS_PATH)
model = model.to(device)
torch.manual_seed(seed=1)

test_dataset  = dataloaders.MilanDataLoader(_set = 'test', toy = False, DATA_DIR=TEST_SET_PATH)
test_data = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,drop_last = True)

_ , y = test_dataset.__getitem__(1)

mse_frame_timestep = np.zeros(y.shape[0])#.to(device)
predictions = []
with tqdm.tqdm(total=len(test_data)) as pbar_test:
    for idx,(x,y) in enumerate(test_data):
        x = x.to(device)
        y = y.to(device)
        out = model.forward(x)
        se_batch = torch.sum((out.squeeze() - y)**2,(2,3))
        mse_frame_timestep = mse_frame_timestep + torch.mean(se_batch,0).cpu().detach().numpy()
        pbar_test.update(1)
        predictions.append(out.cpu().detach().numpy())
### SAVE PREDICTIONS
predictions = np.array(predictions)
np.savez(RESULTS_PATH + RESULT_FOLDERS[model_name] + experiment_name + '/example_predictions/test_predictions.npz',y=predictions)
### SAVE MSE
mse_frame_timestep = mse_frame_timestep / len(test_data)
np.savetxt(RESULTS_PATH + experiment_name + '/mse_frame_timestep.csv', mse_frame_timestep, delimiter=",")
### PLOT MSE/TIMESTEP
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mse_frame_timestep,'-o')
ax.set_title('MSE/frame ' + model_name)
ax.set_xlabel('timestep')
ax.set_ylabel('MSE')
fig.savefig(RESULTS_PATH + 'figures/' + experiment_name + '.pdf')

### SAMPLE PREDICTIONS to create example plots###
x,y = next(iter(test_data))
x = x.to(device)
y = y.to(device)
out = model.forward(x)
out = torch.Tensor.cpu(out)
out = out.detach().numpy()
y = torch.Tensor.cpu(y) # to generate plots

from matplotlib.colors import Normalize
norm = Normalize(vmin=-0.42,vmax=50)
import os

PREDICTIONS_PATH = RESULTS_PATH + RESULT_FOLDERS[model_name] + experiment_name + '/example_predictions/'

colormap = 'nipy_spectral'
if not os.path.isdir(PREDICTIONS_PATH):
    os.mkdir(PREDICTIONS_PATH)
for i in range(y.shape[0]):
    EXAMPLE_PATH = PREDICTIONS_PATH + 'example_' + str(i) 
    if not os.path.isdir(EXAMPLE_PATH):
        os.mkdir(EXAMPLE_PATH)
    for j in range(y.shape[1]):
        
        plt.figure()
        plt.suptitle('timestep: ' + str(j),fontsize=16)

        plt.subplot(1,2,1)

        plt.imshow(y[i,j,...],origin='lower',norm=norm,cmap=colormap)
        plt.title('ground_truth',fontsize=16)
        plt.xlabel('x coordinate',fontsize=16)
        plt.ylabel('y coordinate',fontsize=16)

        plt.subplot(1,2,2)

        plt.imshow(out[i,j,...],origin='lower',norm=norm,cmap=colormap)
        plt.title('prediction',fontsize=16)
        plt.xlabel('x coordinate',fontsize=16)
        # plt.ylabel('y coordinate',fontsize=16)
        plt.subplots_adjust(wspace=0.3)
        fig_name = 'timestep_' + str(j) + '.pdf'
#         plt.colorbar()
        plt.savefig(os.path.join(EXAMPLE_PATH,fig_name))
        plt.clf()
    
    fig,ax =plt.subplots(2,y.shape[1])
    fig.subplots_adjust(wspace = 0.01, hspace=0.01)
    for j in range(y.shape[1]):
        
        fig.suptitle('entire sequence',fontsize=16)
        ax[0,j].imshow(y[i,j,...],origin='lower',norm=norm,cmap=colormap)
        ax[0,j].axis('off')
#         plt.title('ground_truth',fontsize=16)
#         plt.xlabel('x coordinate',fontsize=16)
#         plt.ylabel('y coordinate',fontsize=16)
        ax[1,j].imshow(out[i,j,...],origin='lower',norm=norm,cmap=colormap)
        ax[1,j].axis('off')

#         plt.title('prediction',fontsize=16)
#         plt.xlabel('x coordinate',fontsize=16)
        # plt.ylabel('y coordinate',fontsize=16)
        plt.subplots_adjust(hspace =-.5)
#     fig.text(0.2,0.2,'ground_truth')
#     ax[0,0].set_ylabel('ground_truth',fontsize=16)
#     ax[1,0].set_ylabel('prediction',fontsize=16)

    fig_name = 'entire_sequence' + '.pdf'    
    plt.savefig(os.path.join(EXAMPLE_PATH,fig_name))
    plt.clf()

print('done')