
from data_utils import DataGenerator
import numpy as np 
import pandas as pd
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.applications import VGG16
import csv
from tables import open_file, Atom

# Set a random seed so I can run the same code and get the same result
np.random.seed(7)
        
def feature_generator(batch=30, sample_size=None, 
                      csv_file=None, img_folder=None):
    """
    Convolutional Base Model Architecture:
 ____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
vgg16 (Model)                    (None, 15, 20, 512)   14714688                                     
____________________________________________________________________________________________________
flatten_5 (Flatten)              (None, 153600)        0           block5_pool[0][0]                
====================================================================================================
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
            
    print('initializing data generator...')
    data = DataGenerator() 
    data = data.from_csv(csv_path=csv_file,
                         img_dir=img_folder,
                         batch_size=batch)
    print('initializing model....')
    conv_base = VGG16(include_top=False, 
                      weights='imagenet', 
                      input_shape=(480, 640, 3))

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    
    i = 0
    
    print('Performing Feature Extraction:', end='')
    for inputs, y in data:
        feature_batch = model.predict(inputs)
        print('batch {} through {} complete'.format(i, i + batch))
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
        
                
class array_writer():
    """
    Constructs the pytable storage container for the data, then on 
    each call stores the data in an array
    """
    def __init__(self, chunksize, dataset='HMB_1'):
        self.fileh = open_file('VGG16/{}.h5'.format(dataset), mode = 'w')
        self.root = self.fileh.root
        self.c = chunksize
        
        if dataset not in self.root:
            
            self.HMB1_group = self.fileh.create_group(self.root, dataset)
        
            # feature_table = self.fileh.create_table(HMB1_group, 'features')
            # label_table = self.fileh.create_table(HMB1_group, 'labels')
            
    def __call__(self, arr1, arr2):
        
        a1 = Atom.from_dtype(arr1)
        a2 = Atom.from_dtype(arr2)
            
        if 'test_data1' not in self.HMB1_group:
            self.test_array1 = self.fileh.create_earray(self.HMB1_group,
                                         'test_data1', a1,
                                         shape=(0, len(arr1)),
                                         title='Features Tester',
                                         expectedrows=20,
                                         chunkshape=(self.c, len(arr1)))
    
            self.test_array2 = self.fileh.create_earray(self.HMB1_group,
                                         'test_data2', a2,
                                         shape=(0, len(arr2)),
                                         title='Features Tester',
                                         expectedrows=20,
                                         chunkshape=(self.c, len(arr2)))
        else:
           self.test_array1 = self.root.HMB1_group.test_data1
           self.test_array2 = self.root.HMB1_group.test_data2
            
        self.test_array1.append(arr1)
        self.test_array2.append(arr2)
         

def setup_table(f):
    with fileh as  open_file('VGG16/{}.h5'.format(dataset), mode = 'w')
if __name__ == "__main__":
    
    batch_size = 2
    # Create empty array
    # features = np.zeros(shape=(sample_size, 153600))
    # labels = np.zeros(shape=(sample_size))
    
    VGG = feature_generator(batch=batch_size, sample_size=10)
    
    i = 0
    
    # Store "x" in a chunked array...
    # f = tables.openFile('VGG16/HMB_1.h5', 'w')
    writer = array_writer(batch_size)
#    for feature, label in VGG:
#        writer(feature, label)
        # np.savetxt('features.csv', feature)
#        if 'features' in store:
#            
#            dfx.to_hdf('VGG16/HMB_1.h5', 'features', append=True)
#            dfy.to_hdf('VGG16/HMB_1.h5', 'labels', append=True)
#    
#        else:
#            store.append('features', dfx)
#            store.append('labels', dfy)
#        print('data stored')
    # dfx.to_csv(outfile_features)
    # dfy.to_csv(outfile_labels)
    
#    tfmodel = TransferModel()
#    history = tfmodel.train()
#    
#    acc = history.history['acc']
#    val_acc = history.history['val_acc']
#    loss = history.history['loss']
#    val_loss = history.history['val_loss']
#    
#    epochs = range(len(acc))
#    
#    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
#    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
#    plt.title('Training and Validation accuracy')
#    plt.legend()
#    plt.figure()
#    
#    plt.plot(epochs, loss, 'bo', label='Training loss')
#    plt.plot(epochs, val_loss, 'b', label='Validation loss')
#    plt.title('Training and Validation loss')
#    plt.legend()
#    
#    plt.show()