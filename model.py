import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from datasets.data_utils import load_data_batch
import matplotlib.pyplot as plt

# TO DO: Pre-processing functions for raw data
X_train, Y_train, X_valid, Y_valid = load_data_batch(batch=20)

model = Sequential()

model.add(Conv2D(64, (3, 3),
                 activation='relu',
                 border_mode='same',
                 input_shape=X_train[0].shape))

model.add(Dense(units=1))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=4)  

loss_and_metrics = model.evaluate(X_valid, Y_valid, batch_size=128)


class cnn_model(object):
    
    
    def __init__(self, input_dim=(480, 640, 3), num_filters=53, filter_size=3,
                 hidden_dim=100, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
    
        """
        Initializes a CNN network
        
        Inputs
        ------
        input_dim: Tuple (H, W, C) giving size of input data
        num_filters: Number of filters to use in the convolutional layer
        filter_size: Size of filters to use in the convolutional layer
        hidden_dim: Number of units to use in the affine layer
        weight_scale: Scalar giving standard deviation for random
            initializations of weights.
        reg: Scalar giving L2 regularization strength
        dtype: numpy datatype to use for computation
            
        """
    