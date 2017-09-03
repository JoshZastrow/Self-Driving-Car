import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (20.0, 16.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_data_batch(datasets='HMB_1', batch=50, val_percent=.2,
                    shuffle=False):
    """
    loads in images as features, steering angle as label

    Inputs
    ----
    datasets : subfolder refering to bag folder (i.e 'HMB_ 1')
    batch : size of dataset to be read in
    val_percent: percent of batch to be assigned to validation (i.e 0.2)
    shuffle : TO DO -> shuffle dataset before returning

    returns
    ------
    X_train : (num_train, height, width, channels) array
    Y_train : (num_train, labels) array
    X_valid : (num_valid, height, width, channels) array
    Y_valid : (num_valid, labels) array
    """

    folder = "../Car/datasets/" + datasets + "/output/"
    file = folder + "interpolated.csv"

    # Starting with just center camera
    dataset = pd.read_csv(file)
    dataset = dataset[dataset['frame_id'] == 'center_camera']

    # Add directory path to dataset
    dataset['filename'] = folder + dataset['filename']

    # Setup data placeholders
    assert max(dataset['width']) == min(dataset['width'])
    assert max(dataset['height']) == min(dataset['height'])
    width = max(dataset['width'])
    height = max(dataset['height'])
    channels = 3

    if batch > dataset.shape[0]:
        batch = dataset.shape[0]

    num_train = int(batch * (1 - val_percent))
    num_valid = int(batch * (val_percent))

    X = np.zeros((batch, height, width, channels))
    Y = np.zeros((batch, ))

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

    return X_train, Y_train, X_valid, Y_valid



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
    
    xt, yt, xv, yv = load_data_batch(batch=40, val_percent=.25)
    
    plt.axis('off')
    plt.suptitle("Sample Center Camera Image", fontsize=38)
    plt.imshow(np.uint8(xt[20]))
    print('\nImage Shape:', xt[1].shape)