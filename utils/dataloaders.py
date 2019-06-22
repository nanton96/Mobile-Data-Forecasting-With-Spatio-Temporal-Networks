import numpy as np
import torch.utils.data as data

class MilanDataLoader(data.Dataset):

    def __init__(self,_set='train',toy=False, DATA_DIR = None):
        self.create_channel_axis = create_channel_axis
        if DATA_DIR == None: 
            if toy == True:
                DATA_DIR = 'data_toy/'
            else:
                DATA_DIR = 'data/'
            if _set.lower() == 'train':
                DATA_DIR += 'milan_processed_train.npz'
            elif _set.lower() == 'valid':
                DATA_DIR += 'milan_processed_val.npz'
            elif _set.lower() == 'test':
                DATA_DIR += 'milan_processed_test.npz'
            else:
                raise ValueError("Invalid set. Please select one of: train, valid, test")
        
        data_set = np.load(DATA_DIR)
        # if create_channel_axis == True: 
        #     self.x = np.expand_dims(data_set['x'],4).transpose(0,3,4,1,2).astype(np.float32)   
        #     # DIMENSIONS B S C H W
        #     self.y = np.expand_dims(data_set['y'],4).transpose(0,3,4,1,2).astype(np.float32)   
        #     # seq_len batch_size input_channel height width
        
        self.x = data_set['x'].transpose(0,3,1,2).astype(np.float32)  # DIMENSIONS B S H W
        self.y = data_set['y'].transpose(0,3,1,2).astype(np.float32)

    
    def __getitem__(self,index):
        if self.create_channel_axis == True: 
            # DIMENSIONS S B C H W
            inputs = self.x[index]
            predictions = self.y[index]
        else:
            # DIMENSIONS B S H W
            inputs = self.x[index]
            predictions = self.y[index]
        return inputs, predictions

    def __len__(self):
        return self.x.shape[0]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str