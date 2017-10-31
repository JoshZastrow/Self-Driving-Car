from data_utils import DataGenerator, DataWriter, stopwatch
import numpy as np 
from config import get_user_settings, create_parser
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import csv
from tables import open_file
import os, sys, traceback

# If I go for random sampling... which I don't at the moment.
np.random.seed(7)
        
def feature_generator(batch=30, sample_size=None, 
                      csv_file=None, img_folder=None,
                      model='InceptionV3', start_on_row=0):
    """
    Transfer Learning function.
    Convolutional Base Model Architecture, top layers (classifier) removed
    for low level feature genaration from raw data. Model pulls from a dataset
    and executes a prediction function to generate extracted features.
    
    inputs
    ------
    batch: <int> number of samples to run through the model during prediction
    sample_size: <int> total number of samples to use.
    csv_file: <str> file path to log file. Udacity's file is interpolated.csv
    img_folder: <str> directory path to images. Udacity has three sub folders of images
    model: <str> name of model you would like to use for feature extraction. 
    
    """
    start = time.time()
    # for when I run locally... should be deleted..
    assert os.path.isfile(csv_file), 'CSV log file not found'
    assert os.path.isdir(img_folder), 'image directory not found'
    
    if model == 'InceptionV3':
        from keras.applications import InceptionV3
        print('loading Inception V3 model..')
        conv_base = InceptionV3(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3))
    elif model == 'VGG16':
        from keras.applications import VGG16
        print('loading VGG16 model..')
        conv_base = VGG16(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3)) 
    elif model == 'ResNet50':
        from keras.applications import ResNet50
        print('loading ResNet50 model..')
        conv_base = ResNet50(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3))        
    elif model == 'Xception':
        from keras.applications import Xception
        print('loading Xception model..')
        conv_base = Xception(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3))    
    else:
        model == 'InceptionV3'
    
    stopwatch(start, 'Build Feature Generating Model...')
    model = Sequential()
    model.add(conv_base)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # Get number of lines in the csv file (if not provided)
    if not sample_size:
        with open(csv_file,"r") as f:
            log = csv.reader(f,delimiter = ",")
            log = list(log)
            row = len(log)
    
        if not sample_size or sample_size > row:
            sample_size = row
            
    stopwatch(start, ('\n\nInitializing data generator...'))
    data = DataGenerator() 
    data = data.from_csv(csv_path=csv_file,
                         img_dir=img_folder,
                         batch_size=batch,
                         starting_row=start_on_row)
     
    i = 0
    stopwatch(start, '\nPerforming Feature Extraction:\n')
    
    for inputs, y in data:
        feature_batch = model.predict(inputs)

        print('\tbatch {} through {} complete'
              .format(i, i + batch))

        yield feature_batch, y
        
        i += batch
        if i >= sample_size: break
    

class RegressionModel():
    """
    A basic densely connected Neural Network for regression on extracted features
    from earlier feature extraction models.
    """
    def __init__(self):
    
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
    import time
    start = time.time()
    
    # Get model settings from config file
    config = get_user_settings()
    batch_size = int(config['USER'].get('batch size', 5))
    sample_size = int(config['USER'].get('sample size', 30))
    starting_row = int(config['USER'].get('starting row', 0))
    
    parser = create_parser()
    args = parser.parse_args()
    
    models = ['InceptionV3', 'VGG16', 'Xception', 'ResNet50']
    assert os.path.isfile(args.log), 'Cannot find log file from input path'
    assert os.path.isfile(args.output), 'output file does not exist'
    assert os.path.isdir(args.img_dir), 'image diretory cannot be found'
    assert args.model in models, 'input model is not an option'

    # Timer
    stopwatch(start, 'creating model')
    
    # Use convolutional base model to create features from raw data
    model = feature_generator(batch=batch_size, sample_size=sample_size,
                              csv_file=args.log, 
                              img_folder=args.img_dir,
                              model=args.model,
                              start_on_row=starting_row)
    
    stopwatch(start, 'Model and data loader generated, opening the files')  
    
    with open_file(args.output, mode = 'w') as h5:
        first = True
        i = 0  # row counter
        stopwatch(start, 'starting feature extraction...')
        
        #  Genreate features, write to file
        for output, y in model:
            try:
                i += batch_size
                
                if first: # First loop, create data writer object
                    name = os.path.basename(args.output)
                    name = os.path.splitext(name)[0]
                    store = DataWriter(h5, dataset=name, sample=(output, y))
                    
                store(output, y)
                    
                if not first: continue
                first = False
                stopwatch(start, 'data stored, next batch')     
                
            # If ctrl + c out of the data writer, store the row location
            except KeyboardInterrupt:
                print('keyboard interrupt, exiting program and saving row location')
                config['USER']['starting row'] = str(i)
                with open('config.ini', 'w') as configfile:
                    config.write(configfile)
                sys.exit(0)
                
            except Exception:
                traceback.print_exc(file=sys.stdout)
                sys.exit(0)             

        # Save progress
        config['USER']['starting row'] = str(i)
        with open('config.ini', 'w') as configfile:
            config.write(configfile) 
