
from data_utils import DataGenerator
import matplotlib.pyplot as plt
import numpy as np 
from keras import optimizers
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.applications import VGG16
import csv

# Set a random seed so I can run the same code and get the same result
np.random.seed(7)
        
def feature_generator(batch=30, sample_size=None):
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
             
    =============================================================================
    Total params: 21,611,968
    """
    csv_file = '../Car/datasets/HMB_1/output/interpolated.csv'
    img_folder = '../Car/datasets/HMB_1/output/'
    
    # Get number of lines in the csv file (if not provided)
    if not sample_size:
        with open(csv_file,"r") as f:
            reader = csv.reader(f,delimiter = ",")
            data = list(reader)
            sample_size = len(data)
            
    data = DataGenerator() 
    data = data.from_csv(csv_path=csv_file,
                         img_dir=img_folder,
                         batch_size=32)
    
    model = VGG16(include_top=False, weights='imagenet', )
    
    i = 0
    
    # Create empty array
    features = np.zeros(shape=(sample_size, 15, 20 ,512))
    labels = np.zeros(shape=(sample_size))
    
    for inputs, labels in data:
        feature_batch = model.predict(inputs)
        features[i : i + batch - 1] = feature_batch
        labels[i : i + batch - 1] = labels
        i += batch
        if i >= sample_size: break

    return features
    

class TransferModel():
    
    def __init__(self):
        
        # TODO: Add Conv base 
        self.model = Sequential()
        
        self.model.add(Dense(1024, 
                             activation='relu', 
                             input_shape=(30, 13, 18, 2048)))
        
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
                                       batch_size=30)
        return self.model.fit_generator(
               data_flow,
               samples_per_epoch=200,
               nb_epoch=5,
               verbose=2,
               nb_val_samples=100,
               validation_data=data_flow)
                              
        
if __name__ == "__main__":
    
    features = feature_generator()
    print('feature size:', features.shape)
#    tfmodel = TransferModel()
#    
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