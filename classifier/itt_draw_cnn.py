"""
# prerequisites for tflearn:
# tensorflow/tensorflow-gpu, curses, numpy+mkl, scipy, h5py
# for gpu support: https://www.tensorflow.org/install/install_windows#requirements_to_run_tensorflow_with_gpu_support
# CUDA Toolkit 8.0
# cuDNN v5.1
# CUDA Compute 3.0 or higher GPU
# tensorflow-gpu package
# get necessary wheels from www.lfd.uci.edu/~gohlke/pythonlibs/

ImageGuesserCNN is a convolutional deep neural network
Takes length of category list as argument to get the number of needed nodes in last fully connected layer
Created with the tflearn wrapper for TensorFlow
Input layer expects a grayscale 28x28 pixel array
Uses two convolutional layers, max pooling after each to downsample data
2 fully connected layers to classify image data
Adapted from https://www.tensorflow.org/tutorials/layers

How to use:
Create ITTDrawGuesserCNN instance, giving number of categories to classify as argument
Set epoch and checkpoint path as needed or leave at default values
Train model by providing training data with labels, and test data with labels
Save/load model as needed
Get prediction of category as list of probabilities for each category
"""

import tflearn
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


class ITTDrawGuesserCNN:
    DEFAULT_CHECKPOINT_PATH = 'ITTDrawGuesser.tfl.ckpt'

    # Number of epochs is really low, better 100 or more epochs
    # Number chosen because of time constraints
    # Training and adjusting the neural networks took too much time on CPU, as GPU support couldn't be made to work
    DEFAULT_EPOCH = 10

    # Couldn't make tensorflow-gpu work, recognizes gpu but crashes
    use_cpu_only = True

    def __init__(self, num_categories):
        # Dynamic output node number
        self.num_categories = num_categories

        # training image size is image size from following training data:
        # https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap/?pli=1
        self.training_image_size = (28, 28)
        self.checkpoint_path = self.DEFAULT_CHECKPOINT_PATH
        self.epoch = self.DEFAULT_EPOCH

        self.graph = self._build_graph()
        self.model = self._wrap_graph_in_model(self.graph)

    # Build graph of network
    # Using rectified linear units (relu) as activation function for convolutional layers
    # Using softmax as activation function for output layer
    # todo: experiment with node number in layers, add layers if necessary
    def _build_graph(self):
        # Input: grayscale (one channel) 28x28 image; no image preprocessing
        # Use data_preprocessing argument to process data before going into network
        # Use data_augmentation to create additional synthetic test data
        pre_process_data = tflearn.DataPreprocessing()
        pre_process_data.add_custom_preprocessing(self._reshape_images)
        graph = input_data(shape=[None, 28, 28, 1], data_preprocessing=pre_process_data)

        # Convolutional Layer 1: 32 nodes, 3x3 pixel filter size
        graph = conv_2d(graph, 32, 3, activation='relu', regularizer="L2")

        # Downsampling with 2x2 filter
        graph = max_pool_2d(graph, 2)

        graph = local_response_normalization(graph)

        # Convolutional Layer 2: 64 nodes, 3x3 pixel filter size
        graph = conv_2d(graph, 64, 3, activation='relu', regularizer="L2")

        # Downsampling with 2x2 filter
        graph = max_pool_2d(graph, 2)

        graph = local_response_normalization(graph)

        # Fully connected layer 1, 128 neurons
        graph = fully_connected(graph, 128, activation='relu')

        # Set dropout to 0.5; throwing away random data to prevent over-fitting
        graph = dropout(graph, 0.5)

        # Fully connected layer 2, 256 neurons
        graph = fully_connected(graph, 256, activation='relu')

        # Set dropout to 0.5; throwing away random data to prevent over-fitting
        graph = dropout(graph, 0.5)

        # Fully connected layer 2, 256 neurons
        graph = fully_connected(graph, 512, activation='relu')

        # Set dropout to 0.5; throwing away random data to prevent over-fitting
        graph = dropout(graph, 0.5)

        # Add fully connected layer with number of nodes equal to number of categories to detect
        graph = fully_connected(graph, self.num_categories, activation='softmax')

        # Set optimizing and loss functions for training
        # Loss function: evaluate difference between target category and prediction, adjust weights
        # Cross entropy typical for multi class classification according to https://www.tensorflow.org/tutorials/layers
        # Optimizer: Adaptive Moment Estimation (adam)
        graph = regression(graph, optimizer='adam',
                           loss='categorical_crossentropy',
                           learning_rate=0.001)
        return graph

    def _reshape_images(self, array):
        return array.reshape([-1, 28, 28, 1])

    # Set custom checkpoint filepath
    def set_checkpoint_path(self, filepath):
        self.checkpoint_path = filepath

    # Wrap graph in tflearn DNN model, set the filepath for checkpoint file to save the model after each epoch
    def _wrap_graph_in_model(self, graph):
        model = tflearn.DNN(graph, tensorboard_verbose=0, checkpoint_path=self.checkpoint_path)
        return model

    def set_epoch(self, epoch):
        self.epoch = epoch

    # Trains the Convolutional network graph with:
    # x as training data and y as training categories
    # x_test as validating data and y_test as validating categories
    # Trains epoch number of times, reporting accuracy after each
    # Saves progress after each epoch
    def train(self, x, y, x_test, y_test):

        if self.use_cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            with tf.device('/cpu:0'):
                self.model.fit(x, y, n_epoch=self.epoch, shuffle=True, validation_set=(x_test, y_test),
                               show_metric=True, snapshot_epoch=True)
        else:
            self.model.fit(x, y, n_epoch=self.epoch, shuffle=True, validation_set=(x_test, y_test),
                           show_metric=True, snapshot_epoch=True)

    # Save model to a file after completing training
    def save_model(self, filepath):
        self.model.save(filepath)

    # .tfl File in filepath needed
    def load_model(self, filepath):
        self.model.load(filepath)

    # Takes grayscale image
    # todo: should maybe make sure passed image has correct format of 28x28 grayscale arrray
    def predict(self, image_data_array):
        normalized_image_data = self.normalize_data(image_data_array)
        return self.model.predict(normalized_image_data)

    # Normalize input image array to 28x28 grayscale
    # Find better way to transform numpy.recarray to numpy.array
    def normalize_data(self, image_data):
        if image_data.shape == (self.training_image_size[0]*self.training_image_size[1],):
            return image_data
        else:
            #image_data = np.array(image_data.tolist())
            image = Image.fromarray(image_data, 'RGB')
            image.save('raw.png')
            image = image.convert('L')
            image.save('grayscale.png')
            image = image.resize(self.training_image_size, Image.BILINEAR)
            image.save('resized.png')
            image = np.array(image)

            image = self.boost_non_black_pixels(image)
            print(image)
            return np.array(image)

    def boost_non_black_pixels(self, image):
        for i in range(len(image)):
            for j in range(len(image[0])):
                if image[i][j] != 0:
                    image[i][j] = 255
        image = Image.fromarray(image, 'L')
        image.save('whited.png')
        return np.array(image)
