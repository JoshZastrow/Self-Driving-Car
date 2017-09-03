
from datasets.data_utils import load_data_batch
import matplotlib.pyplot as plt
import numpy as np 
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D
from keras.layers import AveragePooling2D, ZeroPadding2D, Dropout
from keras.layers import Flatten, merge, Reshape, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras.applications.inception_v3 import InceptionV3
from layers import PoolHelper,LRN
import scipy.misc

# Set a random seed so I can run the same code and get the same result
np.random.seed(7)
    
def transfer_InceptionV3():
    
    
    # Training Parameters
#    img_width, img_height = 640, 480
#    train_data_dir = "~//Documents//Code//Car//datasets//" \
#                     "HMB_1//output//center"
#    validation_data_dir = "data/val"
#    nb_train_samples = 4125
#    nb_validation_samples = 466 
#    batch_size = 16
#    epochs = 50

    # Create
    base_model = InceptionV3(include_top=False, 
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(480, 640, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
        
    top_layers = base_model.output
    top_layers = GlobalAveragePooling2D()(top_layers)
    top_layers = Dense(1024, activation='relu')(top_layers)
    prediction = Dense(1)(top_layers)
   
    model = Model(inputs=base_model.input,
                  outputs=prediction)    
    model.summary()
    
    # Compile
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    # Train
    X_train, Y_train, X_test, Y_test = load_data_batch()
    model.fit(x=X_train, y=Y_train)
    

if __name__ == "__main__":
    
    img_file = r"C:\Users\Joshu\Documents\Code\Car\datasets\HMB_1\output\center\1479424331254463377.png"
    img = scipy.misc.imread(img_file, mode='RGB') #.astype(np.float32)
    plt.imshow(img)
    # img = scipy.misc.imresize(scipy.misc.imread(img_file, mode='RGB'), (299, 299)).astype(np.float32)
    
    # img[:, :, 0] -= 123.68
    # img[:, :, 1] -= 116.779
    # img[:, :, 2] -= 103.939

    # img[:,:,[0,1,2]] = img[:,:,[2,1,0]]

    # img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    x = transfer_InceptionV3()
    
    
    # Test googlenet model
#    model = create_googlenet('googlenet_weights.h5')
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='categorical_crossentropy')
#    out = model.predict(img) # note: the model has three outputs
#    print(np.argmax(out[2]))
    