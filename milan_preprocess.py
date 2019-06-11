
import pandas as pd
import numpy as np
import random
import os
import glob
import argparse
seed = 12345
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(
        description='Welcome to Milan dataset preprocess script. This script generates the data necessary for supervised learning from the raw Milan traffic data.')
    parser.add_argument('--S',type = int, default = 5,help='specifies number of observations')
    parser.add_argument('--K',type = int, default = 15,help='specifies number of predictions')
    parser.add_argument('--shifted_predictions',type = bool, default = False,help='if true then we produce targets of shifted data, else we produce targets of the data directly after the observations')
    
    args = parser.parse_args()
    process_milan_dataset(args.S,args.K,args.shifted_predictions)

def process_milan_dataset(S=12,K=4,shift_flag=True):
    '''
    This function will return a tensorflow Dataset consisting of tensors from the milan dataset.
    N is the number of training examples
    The features consist of a Nx100x100xS tensor.
    Predictions (labels) consist of a Nx100x100xK tensor.

    params:
    int S : specifies number of observations
    int K : specifies number of predictions

    outputs: 
        .npz files containing a training, validation and test set in the form of numpy arrays
        each file contains two arrays:
        x : features (inputs) (Nx100x100xS tensor)
        y : predictions (labels) (Nx100x100xK tensor)
    '''

    DATA_PATH = 'data'
    SAVE_FILE = 'data/milan_processed'
    all_files = glob.glob(os.path.join(DATA_PATH, "*.txt")) 

    # load files
    df_from_each_file = (pd.read_csv(f,sep='\t',header=None,usecols=[0,1,2,6],
                                    names=['Square id','TimeInterval','Country Code','Traffic']) \
                        for f in all_files)
    df = pd.concat(df_from_each_file, ignore_index=True)
    # preprocess dataset
    df = milan_preprocess(df)
    # convert to numpy feature and label tensors
    x,y = dataframe_to_numpy_arrays(df,S,K,shift_flag)
    x,y = shuffle(x,y,random_state = seed)
    # transform to [1,-1] range
    #### NORMALISATION #####
    #max_x = np.max(x[...])
    #x = 2 * (x / max_x) - 1
    #y = 2 * (y / max_x) - 1
    #### STANDARDISATION ###
    mean_x = np.mean(x.flatten())
    std_x  = np.std(x.flatten())
    x = (x - mean_x) / std_x
    y = (y - mean_x) / std_x
    # split to train,val, test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    np.savez(SAVE_FILE+'_train.npz',x=x_train,y=y_train)
    np.savez(SAVE_FILE+'_val.npz',x=x_val,y=y_val)
    np.savez(SAVE_FILE+'_test.npz',x=x_test,y=y_test)

def milan_preprocess(df):
    '''
    This function applies some necessary preprocessing to convert the dataframe to one suitable for training.

    '''
    #convert unix timecode to dates
    df['Time interval'] = pd.to_datetime(df['TimeInterval'],unit='ms')
    #drop NaN values
    df.dropna(inplace=True)
    #Sum rows that are on the same square and time
    #ie marginalize over Country Code
    df = df.groupby(['Square id','TimeInterval'],as_index=False)['Traffic'].sum()

    df['x'] = (df['Square id']-1) % 100 
    df['y'] = (df['Square id']-1) // 100
    df['t'] = (df['TimeInterval']) // 600000 - 2305434
    df.drop('Square id',inplace=True,axis=1)
    df.drop('TimeInterval',inplace=True,axis=1)
    df = df[['x','y','t','Traffic']]
    
    return df

def dataframe_to_numpy_arrays(df,S,K,shift_flag):

    '''
    Converts the dataframe to numpy arrays.

    params:
    pd.dataframe df: Milan dataframe 
    
    int S : specifies number of observations
    int K : specifies number of predictions

    outputs:
    np.array x : tensor of features    shape (Nx100x100xS)
    np.array y : tensor of predictions shape (Nx100x100xK)


    '''
    #transform data to 3D numpy array (x,y,t)
    #ref: https://stackoverflow.com/questions/47715300/convert-a-pandas-dataframe-to-a-multidimensional-ndarray
    grouped = df.groupby(['x','y','t']).mean()
    shape = tuple(map(len,grouped.index.levels))
    raw= np.zeros(shape)
    raw[grouped.index.labels] = grouped.values.flat

    L = raw.shape[2]

    x_t = [raw[:,:,t:t+S] for t in range(L-S-K)]
    if shift_flag == True: 
        y_t = [raw[:,:,t+1:t+S+1] for t in range(L-S-K)]
    else:
        y_t = [raw[:,:,t+S:t+S+K] for t in range(L-S-K)]
    x = np.array(x_t)
    y = np.array(y_t)

    return x,y

if __name__ == "__main__":
    main()