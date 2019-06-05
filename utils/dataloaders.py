import numpy as np

class MilanDataLoader():

    def __init__(self):

        DATA_DIR = 'data_toy/milan_processed_train.npz'
        data_set = np.load(DATA_DIR)
        # self.x = np.expand_dims(data_set['x'],4).transpose(0,3,1,2,4)
        # self.y = np.expand_dims(data_set['y'],4).transpose(0,3,1,2,4)
        self.x = data_set['x'].transpose(0,3,1,2).astype(np.float32)
        self.y = data_set['y'].transpose(0,3,1,2).astype(np.float32)
        print(self.x.shape[0])
    
    def __getitem__(self,index):

        inputs = self.x[index]
        predictions = self.y[index]

        return inputs, predictions

    def __len__(self):
        return self.x.shape[0]