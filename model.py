from data_utils import DataGenerator, DataWriter
import numpy as np 
import argparse
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.applications import InceptionV3, VGG16, ResNet50, Xception
import csv
from tables import open_file
import os

# If I go for random sampling... which I don't at the moment.
np.random.seed(7)
        
def feature_generator(batch=30, sample_size=None, 
                      csv_file=None, img_folder=None,
                      model='InceptionV3'):
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
    
    # for when I run locally... should be deleted..
    if not csv_file:
        csv_file = '../Car/datasets/HMB_1/output/interpolated.csv'
        img_folder = '../Car/datasets/HMB_1/output/'
    
    # Get number of lines in the csv file (if not provided)
    if not sample_size:
    with open(csv_file,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        rows = len(data)
    
    if not sample_size or sample_size > rows:
        sample_size = rows
            
    print('initializing data generator...', end='')
    data = DataGenerator() 
    data = data.from_csv(csv_path=csv_file,
                         img_dir=img_folder,
                         batch_size=batch)

    if model == 'InceptionV3':
        conv_base = InceptionV3(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3))
    elif model == 'VGG16':
        conv_base = VGG16(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3)) 
    elif model == 'ResNet50':
        conv_base = ResNet50(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3))        
    elif model == 'Xception':
        conv_base = Xception(include_top=False, 
                                weights='imagenet', 
                                input_shape=(480, 640, 3))    
    else:
        model == 'InceptionV3'
        
    model = Sequential()
    model.add(conv_base)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    i = 0
    
    print('\nPerforming Feature Extraction:\n')
    
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
    
    parser = argparse.ArgumentParser(description='Executes a '
                                                 'state of the art ML model '
                                                 'for feature extraction')
    parser.add_argument('-log', metavar='csv_file', type=str, nargs='?',
                        required=True,
                        help='A file path to the CSV log file holding the '
                             'sensor data.\n Example: data/interpolated.csv')
    parser.add_argument('-img_dir', metavar='image_folder', type=str, nargs='?',
                        required=True,
                        help='path to the directory containing the image folders'
                             '\n\nNOTE: This directory should have the center,'
                             ' left, and right image subfolders as referenced'
                             ' in the log file. This directory will be the'
                             ' base path for the image file paths listed'
                             ' in the log file. Example: data/images')
    parser.add_argument('-model', type=str, nargs='?',
                        help='the model you would like to use for feature'
                             ' extraction. Current options are:'
                             '\t-InceptionV3\n'
                             '\t-VGG16\n'
                             '\t-ResNet50\n'
                             '\t-Xception'
                             '\n\nleaving this blank will default to InceptionV3')
    parser.add_argument('-output', type=str, nargs='?',
                        required=True,
                        help='Output file path. Must be an .h5 file'
                             '\n\nExample:\n\t'
                             'outputs/InceptionV3.h5')
    
    args = parser.parse_args()
    
    assert os.path.isfile(args.log), 'Cannot find log file from input path'
    assert os.path.isfile(args.output), 'output file does not exist'
    assert os.path.isdir(args.img_dir), 'image diretory cannot be found'
    assert args.model in ['InceptionV3', 'VGG16'], 'input model is not an option'
    
    # Pulls data from the dataset and runs a prediction
    # The prediction is a semi processed batch of data.
    # the output of this processing is paired with the ground truth of the data.
    model = feature_generator(batch=5, sample_size=10,
                              csv_file=args.log, 
                              img_folder=args.img_dir,
                              model=args.model)

    with open_file(args.output, mode = 'w') as h5:
        first = True
        
        for output, y in model:
            if first:
                name = os.path.basename(args.output)
                name = os.splitext(name)[0]
                store = DataWriter(h5, dataset=name, sample=(output, y))

            store(output, y)
            first = False

    
