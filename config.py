# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:54:35 2017

@author: joshua
"""

import configparser
import argparse

def get_user_settings():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config
    
def create_parser():
    parser = argparse.ArgumentParser(description='Executes a '
                                                 'state of the art ML model '
                                                 'for feature extraction')
    
    parser.add_argument('-log', metavar='csv_file', type=str, nargs='?',
                        required=True,
                        help='A file path to the CSV log file holding the '
                             'sensor data.\n Example: data/interpolated.csv')
    
    parser.add_argument('-img_dir', type=str, nargs='?',
                        required=True,
                        help='path to the directory containing the image folders'
                             '\n\nNOTE: This directory should have the center,'
                             ' left, and right image subfolders as referenced'
                             ' in the log file. This directory will be the'
                             ' base path for the image file paths listed'
                             ' in the log file. Example: data/images')
    
    parser.add_argument('-model', type=str, nargs='?',
                        required=False,
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
    
    return parser


if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    
    config['DEFAULT'] = {
            'batch size': 5,
            'sample size': 20,
            'frame height': 480,
            'frame width': 640,
            'starting row': 0,
            } 
    
    config['USER'] = {}
    
    batch = input('<int> Enter number of images to read per batch process --> ')
    sample_size = input('<int> Enter total number of images to process --> ')
    starting_row = input('<int> Enter row of log file to start on (Default is 0) --> ')
    
    print('\nThe following configurations are optional, press enter to skip.'
          '\nThe default frame size is 480 x 640\n')
    
    frame_height = input('<int> Enter frame height of image sample --> ')
    frame_width = input('<int> Enter frame width of image sample -->')
    
    if batch:
        config['USER']['batch size'] = batch
    else:
        config['USER']['batch size'] = config['DEFAULT']['batch size']
        
    if sample_size:
        config['USER']['sample size'] = sample_size
    else:
        config['USER']['sample size'] = config['DEFAULT']['sample size'] 
        
    if starting_row:
        config['USER']['starting row'] = starting_row
    else:
        config['USER']['starting row'] = config['DEFAULT']['starting row']
 
    if frame_height:
        config['USER']['frame height'] = frame_height
    else:
        config['USER']['frame height'] = config['DEFAULT']['frame height']  
             
    if frame_width:
        config['USER']['frame width'] = frame_width
    else:
        config['USER']['frame width'] = config['DEFAULT']['frame width']
             
    with open('config.ini', 'w') as configfile:
            config.write(configfile)
