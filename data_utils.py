from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os, math, time
from tables import Atom

plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class DataGenerator(ImageDataGenerator):
    """
    A data generator object that flows data from selected source.
    Initializes with parameters from Keras ImageDataGenerator.
    """
    def __init__(self, *args, **kwargs):
        ImageDataGenerator.__init__(self, *args, **kwargs)
        self.iterator=None
    
    def from_csv(self, 
                 csv_path,
                 img_dir, 
                 batch_size, 
                 target_size=(480, 640),
                 col_headers=['angle'],
                 starting_row=0):

        assert os.path.isfile(csv_path), 'CSV Log file cannot be found'
        assert os.path.isdir(img_dir), 'Image directory cannot be found'
        
        # CSV Stores labels and filepath to image
        reader = pd.read_csv(csv_path, 
                             chunksize=batch_size, 
                             skiprows=starting_row)
        
        # Yield one set of images 
        for batch in reader:
            data = self.process_images(batch['filename'], target_size)
            labels = np.array(batch[col_headers])
            
            yield data, labels

    def process_images(self, dir_list, target_size, img_dir=None):
        """
        Loads images from file, performs image processing (if any)
        
        inputs
        ------
        dir_list: list of image file paths
        target_size: desired size of images
        
        returns
        -------
        images: np array of images
        """
        
        img_dir = '../Car/datasets/HMB_1/output/' if not img_dir else img_dir
        dir_list = img_dir + dir_list
        batch_size = len(dir_list)
        
        images = np.zeros(shape=(batch_size, *target_size, 3))
        
        for i, line in enumerate(dir_list):
            get_image = misc.imread(line, mode='RGB')
            
            # TODO: Figure out how to use the image processing features
            #       of the inherited DataGenerator on the loaded image
            images[i] = get_image
            
        return images
    

class DataWriter(object):
    """
    writes numpy array to .h5 file. Creates a dataset group, then 
    a feature and label subgroup which stores each array named with their
    original filename.
    """
    def __init__(self, fileh, dataset, sample):
        root = fileh.root
        a1 = Atom.from_dtype(sample[0].dtype)  # feature data shape
        a2 = Atom.from_dtype(sample[1].dtype)  # label data shape
        batch, size = sample[0].shape
        
        if dataset not in fileh.root:
            group = fileh.create_group(
                    root, 
                    name=dataset,
                    title='{} dataset'.format(dataset))
        else:
            group = fileh.get_node(root, dataset)
            
        if 'features' not in group:
            self.features = fileh.create_earray(
                    where=group,
                    name='features',
                    atom=a1,
                    shape=(0, size),
                    title='feature dataset', 
                    chunkshape=(batch, size))
        else:
            self.features = fileh.get_node(group, 'features')
            
        if 'labels' not in group:
            self.labels = fileh.create_earray(
                    where=group,
                    name='labels',
                    atom=a2,
                    shape=(0, 1),
                    title='label dataset',
                    chunkshape=(batch, 1))                        
        else:
            self.labels = fileh.get_node(group, 'labels')
            
    def __call__(self, x, y):
        self.features.append(x)
        self.labels.append(y)
        
 
def load_udacity_data(file_path='', 
                      img_dir='', 
                      batch=100, val_percent=.2,
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
       
if __name__ == "__main__":
    
    file_loc = '../Car/datasets/HMB_1/output/interpolated.csv'
    i = 0
        
    f = 1
    reader = DataGenerator(samplewise_center=True)
    reader = reader.from_csv(csv_path=file_loc,
                             img_dir='../Car/datasets/HMB_1/output/',
                             batch_size=30)
    c = 5
    print('Data Utilities, reading sample images from HMB_1 datastet..\n\n')
    for chunk in reader:
        if f > c: break
        f += 1
        fig, ax = plt.subplots(nrows=1, ncols=c, sharex=True, squeeze=True)
        j = 0
        
        for axis in ax:        
            axis.set_title(chunk[1][j])
            axis.imshow(np.uint8(chunk[0][j]))
            axis.axis('off')
            j += 1
            
        for i in range(c):
            plt.subplot('15{}'.format(i + 1))
            plt.imshow(np.uint8(chunk[0][i]))
            plt.title = 'Steering Angle {}: {}'.format(i, chunk[1][i])
            plt.axis('off')
        plt.show()

def stopwatch(start, comment):
    lap = math.floor(time.time() - start)
    print('{}: {}:{} sec'.format(comment, lap // 60, lap % 60))
    