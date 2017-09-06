
from data_utils import load_data_batch
import matplotlib.pyplot as plt
import numpy as np 
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

# Set a random seed so I can run the same code and get the same result
np.random.seed(7)
    
def transfer_InceptionV3():
    """
    Model Architecture:
    _____________________________________________________________________________
    Layer (type)         Output Shape          Param #  Connected to                     
    =============================================================================
    input_6              (None, 480, 640, 3)   0                                            
    (InputLayer)
    _____________________________________________________________________________
    InceptionV3          (None, 13, 18, 2048)  896      input_6[0][0]
    (Tranfer Model)
    _____________________________________________________________________________
    globalavepooling2d_6 (None, 2048)          0        InceptionV3[0][0]   
    (Pool)                 
    _____________________________________________________________________________
    dense_9              (None, 1024)          2098176  globalavepooling2d_6[0][0]
    (Dense) 
    _____________________________________________________________________________
    dense_10             (None, 1)             1025     dense_9[0][0]      
    (Dense)              
    =============================================================================
    Total params: 23,711,169
    Trainable params: 2,099,201
    Non-trainable params: 21,611,968
    """
    
    # Create
    base_model = InceptionV3(include_top=False, 
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(480, 640, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add
    top_layers = base_model.output
    top_layers = GlobalAveragePooling2D()(top_layers)
    top_layers = Dense(1024, activation='relu')(top_layers)
    prediction = Dense(1)(top_layers)
   
    model = Model(input=base_model.input,
                  output=prediction)    
    
    # Compile: Regression 
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    return model
        
if __name__ == "__main__":

    model = transfer_InceptionV3()
    
    # Load Data
    X_train, Y_train, X_test, Y_test = load_data_batch()
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    
    
    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    model.fit(x=X_train, y=Y_train, epoch=3)
    
    score = model.evaluate(x=X_test, y=Y_test)
    