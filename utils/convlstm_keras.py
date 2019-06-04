import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

def main():
    DATA_DIR = 'data_toy/milan_processed_train.npz'
    data_set = np.load(DATA_DIR)
    x = np.expand_dims(data_set['x'],4).transpose(0,3,1,2,4)
    y = np.expand_dims(data_set['y'],4).transpose(0,3,1,2,4)
    n_pixel = 100

    seq = Sequential()
    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       input_shape=(None, n_pixel, n_pixel, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    seq.compile(loss='mean_squared_error', optimizer='adadelta')

    # Train the network
    seq.fit(x, y, batch_size=10,
            epochs=50, validation_split=0.05,verbose=1)

    #save model
    seq.save('ConvLSTM.h5')

if __name__ == '__main__':
    main()
