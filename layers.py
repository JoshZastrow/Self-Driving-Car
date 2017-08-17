# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 11:26:16 2017

@author: joshua zastrow

Note: Shape of Images are (num_train, height, width, channels)
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Convolution_layer(Layer):

    """
    A keras implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C chs, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C chs and has height HH and width HH.



    Input:
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel

    - conv_param: A dictionary with the following keys:

        - 'stride': The number of pixels between adjacent receptive fields in the

           horizontal and vertical directions.

        - 'pad': The number of pixels that will be used to zero-pad the input.



    Returns a tuple of:

    - out: Output data, of shape (N, F, H', W') where H' and W' are given by

        H' = 1 + (H + 2 * pad - HH) / stride

        W' = 1 + (W + 2 * pad - WW) / stride


    """
    def __init__(self, kernel_height=3, kernel_width=3, 
                 nb_filters, stride=None, pad=None,
                 **kwargs):

        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.nb_filters = nb_filters
        self.stride = stride
        self.pad = pad
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        
        # Create a trainable weight variable
        height, width, channels = input_shape[1]
        self.W_shape = (self.nb_filters, 
                        channels, 
                        self.kernel_height, 
                        self.kernel_width)
        
        self.W = self.add_weight(shape=self.W_shape,
                                      name='{}_W'.format(self.name),
                                      initializer='uniform',
                                      trainable=True)
        
        self.b = self.add_weight((self.nb_filters), 
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer
                                 
        super().build(input_shape)
        
        
    def call(self, x):
        
        x = K.expand_dims(x, 2)  # adds dummy dimension
        
        conv = K.conv2d(x, self.W, 
                          strides=self.stride, 
                          border_mode='valid',
                          dim_ordering='tf')
        
        conv = K.squeeze(conv, 2)  # removes dummy dimension
        conv += K.reshape(self.b, (1, 1, self.nb_filters))
        
        conv_relu = K.relu(conv, alpha=0., max_value=None)
        conv_relu_pool = K.pool2d(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        length = conv_output_length(input_shape[1],
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0])
        return (input_shape[0], length, self.nb_filter)
    
class Fully_Connected_layer(Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='{}_w'.format(name),
                                 shape=(input_shape[1], self.output_dim),
                                 trainable=True)
        
        self.b = self.add_weight(name='{}_b'.format(name),
                                 shape=(input_shape[1], 1))
        
        super().build(self, input_shape)
        
    def call(self, x):
        
        output = K.dot(x, self.W)
        output += self.b
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
                                 
                                
                                
        
        
        