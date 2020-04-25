import os
import shutil
import tensorflow as tf
import numpy as np
from math import *
import copy


def get_cell_param(parameter):

    param = {}
    param['input_channels'] = parameter[0]
    param['output_channels'] = parameter[1]
    param['input_to_state_kernel_size'] = (parameter[2],parameter[2])
    param['state_to_state_kernel_size'] = (parameter[3],parameter[3])
    return param

def get_pool_param(parameter,mode = 'max',padding = 'SAME'):

    param = {}
    param['padding'] = padding
    param['pool_mode'] = mode
    param['pool_size'] = (1,parameter[0],parameter[0],1)
    param['strides'] = (1,parameter[1],parameter[1],1)

    return param

def normalization(frames):
    new_frames = frames.astype(np.float32)
    new_frames /= (80/2)
    new_frames -= 1
    return new_frames

def denormalization(frames):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (80/2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def get_conv_param(parameter,padding='SAME',activate='relu'):
    param = {}
    param['input_channel'] = parameter[0]
    param['output_channel'] = parameter[1]
    param['kernel_size']=(parameter[2],parameter[2])
    param['padding'] = padding
    param['activate'] = activate
    return param

def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def weight_variable(shape):
    return tf.get_variable("w", shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype= tf.float32)

def bias_variable(shape):
    return tf.get_variable("b", shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype= tf.float32)

def batch_norm(input):
    axis = list(range(len(input.get_shape().as_list()) - 1))  # len(x.get_shape())
    a_mean, a_var = tf.nn.moments(input, axis)
    return tf.nn.batch_normalization(
        input,
        a_mean,
        a_var,
        None,
        None,
        1e-05,
    )




def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv3d_new(x, W, b, pad_type = 'SAME', test_input=None, active='elu',):
    conv_output = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding=pad_type)
    if test_input is not None:
        test_conv_output = tf.nn.conv3d(
            test_input,
            W,
            strides=[1, 1, 1, 1, 1],
            padding=pad_type
        )
        if active == 'elu':
            return tf.nn.elu(conv_output+b), tf.nn.elu(test_conv_output+b)
        if active == 'relu':
            return tf.nn.relu(conv_output+b), tf.nn.relu(test_conv_output+b)
        if active == 'sigmoid':
            return tf.nn.sigmoid(conv_output+b), tf.nn.sigmoid(test_conv_output+b)
    else:
        if active == 'elu':
            return tf.nn.elu(conv_output+b)
        if active == 'relu':
            return tf.nn.relu(conv_output+b)
        if active == 'sigmoid':
            return tf.nn.sigmoid(conv_output+b)

def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv3d(x, W, pad_type = 'SAME'):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding=pad_type)

def max_pool_3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 2, 2, 2, 1], padding='SAME')


def calc_outshape(in_height, in_width, strides, type):
    if type == 'SAME':
        out_height = ceil(float(in_height)) / float(strides[2])

