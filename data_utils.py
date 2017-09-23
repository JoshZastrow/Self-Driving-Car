from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os
import matplotlib.cbook as cbook

plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_udacity_data(file_path='', img_dir='', batch=100, val_percent=.2,
                    shuffle=False, rescale=True):
    """
    loads in images as features, steering angle as label

    Inputs
    ----
    datasets : subfolder refering to bag folder (i.e 'HMB_ 1')
    batch : total number of samples to be read in
    val_percent: percent of batch to be assigned to validation (i.e 0.2)
    shuffle : TO DO -> shuffle dataset before returning

    returns
    ------
    X_train : (num_train, height, width, channels) array
    Y_train : (num_train, labels) array
    X_valid : (num_valid, height, width, channels) array
    Y_valid : (num_valid, labels) array
    """

    # Dataset folder
    if not file_path:
        file = "../Car/datasets/HMB_1/output/interpolated.csv"
        assert os.path.isfile(file), 'interpolated dataset not found'
    
    if not img_dir:
        img_dir = '../Car/datasets/HMB_1/output/'
        assert os.path.isdir(img_dir)
        
    # Starting with just center camera
    dataset = pd.read_csv(file)
    dataset = dataset[dataset['frame_id'] == 'center_camera']

    # Add directory path to dataset
    dataset['filename'] = img_dir + dataset['filename']

    # Setup data placeholders
    assert max(dataset['width']) == min(dataset['width'])
    assert max(dataset['height']) == min(dataset['height'])
    
    width = max(dataset['width'])
    height = max(dataset['height'])
    channels = 3

    if batch > dataset.shape[0]:
        batch = dataset.shape[0]

    X = np.zeros((batch, height, width, channels))
    Y = np.zeros((batch, ))

    num_train = int(batch * (1 - val_percent))
    num_valid = int(batch * (val_percent))
    
    mask = range(num_train, num_train + num_valid)
    X_valid = X[mask]
    Y_valid = Y[mask]
    
    mask = range(num_train)
    X_train = X[mask]
    Y_train = Y[mask]

    del X
    del Y

    count = 0

    # read in file data
    for rw in range(0, batch):

        angle = dataset['angle'].iloc[rw]
        ipath = dataset['filename'].iloc[rw]
        image = misc.imread(ipath)

        if count < num_train:
            X_train[count] = image
            Y_train[count] = angle
        else:
            X_valid[count % num_train] = image
            Y_valid[count % num_train] = angle

        count += 1

    data = {'X_train': X_train,
            'Y_train': Y_train,
            'X_valid': X_valid,
            'Y_valid': Y_valid}
    
    if rescale:
        data['X_train'] /= 255
        data['X_valid'] /= 255
        
    return data



def load_commai_data(log_file, cam_file):
    """
    loads .h5 files from comma AI's car dataset.

    Inputs
    ----
        log_file: file path for sensor log .h5 file
        cam_file: camera path for camera frames from .h5 file

    Returns
    -------
        log: Pandas Dataframe of log file, indexed with cam1_ptr
        cam: PyTables CArray of shape (frame, height, width, channels))

    """
    log_store = pd.HDFStore(log_file)
    cam_store = pd.HDFStore(cam_file)

    samples = len(log_store.root.cam1_ptr1)
    data_dic = {}

    # Read datasets into dictionary
    for d in log_store.root:
        if d.shape[0] == samples:
            if d.ndim == 1:
                data_dic[d.name] = d[:]
            else:
                for dim in range(d.shape[1]):
                    data_dic['{}-{}'.format(d.name, dim + 1)] = d[:, dim]

    # Average the log sensors in a Dataframe, create cam 4D array
    log = pd.DataFrame(data_dic).groupby('cam1_ptr').mean()
    cam = cam_store.root.X[:]

    return log, cam


class DataGenerator(ImageDataGenerator):
    """
    A data generator object that flows data from selected source.
    Initializes with parameters from Keras ImageDataGenerator.
    """
    def __init__(self, *args, **kwargs):
        ImageDataGenerator.__init__(self, *args, **kwargs)
        self.iterator=None
    
    def flow_from_csv(self, 
                      csv_path=None,
                      img_dir='', 
                      batch_size=5, 
                      target_size=(480, 640),
                      col_headers=['angle']):
        
        assert os.path.isfile(csv_path), 'Log file cannot be found'
        assert os.path.isdir(img_dir), 'img directory cannot be found'
        
        # CSV Stores labels and filepath to image
        reader = pd.read_csv(csv_path, 
                             chunksize=batch_size)
        
        # Yield one set of images 
        for batch in reader:
            img_path = img_dir + batch['filename']
            data = process_images(img_path, target_size, batch_size)
                
            labels = np.array([batch[h] for h in col_headers])
            labels = labels.transpose(1, 0)
            
            yield data, labels


def process_images(dir_list, target_size, batch_size):
    
    images = np.zeros(shape=(batch_size, *target_size, 3))
    
    for i, line in enumerate(dir_list):
        get_image = misc.imread(line, mode='RGB')
        images[i] = get_image  #.resize(*target_size, 3)
        
    return images
    
if __name__ == "__main__":
    
    f = 1
    reader = DataGenerator()
    reader = reader.flow_from_csv(csv_path='../Car/datasets/HMB_1/output/interpolated.csv',
                                 img_dir='../Car/datasets/HMB_1/output/')

    for chunk in reader:
        if f > 2: break
        f += 1  
      
        fig, ax = plt.subplots(nrows=1, ncols=5, sharex=True, squeeze=True)
        j = 0
        for axis in ax:        
            axis.set_title(chunk[1][j])
            axis.imshow(np.uint8(chunk[0][j]))
            axis.axis('off')
            j += 1
            
        for i in range(5):
            plt.subplot('15{}'.format(i + 1))
            plt.imshow(np.uint8(chunk[0][i]))
            plt.title = 'Steering Angle {}: {}'.format(i, chunk[1][i])
            plt.axis('off')
        plt.show()
#    ax.axis('off')  # clear x- and y-axes
#    plt.show()