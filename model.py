
from data_utils import DataGenerator, DataWriter
import numpy as np 
import pandas as pd
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.applications import VGG16, InceptionV3
import csv
import time
from tables import open_file

# Set a random seed so I can run the same code and get the same result
np.random.seed(7)
        
def feature_generator(batch=30, sample_size=None, 
                      csv_file=None, img_folder=None):
    """
    Convolutional Base Model Architecture:
 ______________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
===============================================================================
vgg16 (Model)                    (None, 15, 20, 512)   14714688                                     
________________________________________________________________________________
flatten_5 (Flatten)              (None, 153600)        0           block5_pool[0][0]                
===============================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    """
    if not csv_file:
        csv_file = '../Car/datasets/HMB_1/output/interpolated.csv'
        img_folder = '../Car/datasets/HMB_1/output/'
    
    # Get number of lines in the csv file (if not provided)
    if not sample_size:
        with open(csv_file,"r") as f:
            reader = csv.reader(f,delimiter = ",")
            data = list(reader)
            sample_size = len(data)
            
    print('initializing data generator...', end='')
    data = DataGenerator() 
    data = data.from_csv(csv_path=csv_file,
                         img_dir=img_folder,
                         batch_size=batch)
    
    t1 = time.time()
    print('{} seconds\ninitializing model... '
          .format(round(t1 - t0, 2)), end='')
    
    conv_base = InceptionV3(include_top=False, 
                            weights='imagenet', 
                            input_shape=(480, 640, 3))
    
    model = Sequential()
    model.add(conv_base)
    model.add(MaxPooling2D(pool_size=(3, 4)))
    model.add(Flatten())
    
    i = 0
    t2 = time.time()
    
    print('{} seconds\n'
          .format(round(t2 - t1, 2)))
    print('\nModel Summary')
    print('-------------')
    print(model.summary())
    print('\nPerforming Feature Extraction:\n')
    
    for inputs, y in data:
        cur = time.time()
        feature_batch = model.predict(inputs)
        new = time.time()
        print('\tbatch {} through {} complete ({} sec)'
              .format(i, i + batch, round(new - cur, 2)))

        yield feature_batch, y
        
        i += batch
        if i >= sample_size: break
    

class TransferModel():
    
    def __init__(self):
        
        # TODO: Add Conv base 
        self.model = Sequential()
        
        self.model.add(Dense(1024, 
                             activation='relu', 
                             input_shape=(5, 13, 18, 2048)))
        
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(1))
        
        self.model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                           loss='mean_squared_error',
                           metrics=['accuracy'])
        
    def train(self, csv_file=None, img_dir=None):
        
        if not (csv_file or img_dir):
            csv_file = '../Car/datasets/HMB_1/output/interpolated.csv'
            img_folder = '../Car/datasets/HMB_1/output/'
            
        data_flow = DataGenerator()
        data_flow = data_flow.from_csv(csv_path=csv_file,
                                       img_dir=img_folder, 
                                       batch_size=5)
        return self.model.fit_generator(
               data_flow,
               samples_per_epoch=200,
               nb_epoch=5,
               verbose=2,
               nb_val_samples=100,
               validation_data=data_flow)
            
        
if __name__ == "__main__":
    
    t0 = time.time()
    
    Inception = feature_generator(batch=5, sample_size=15)
    
    with open_file('Inception/InceptionV3.h5', mode = 'w') as h5:

        i = 0
        for x, y in Inception:
            if i > 4: break
            if i == 0:
                w = DataWriter(h5, dataset='Test', sample=(x, y))
            w(x, y)
            
            i += 1

    
