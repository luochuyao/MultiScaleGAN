import tensorflow as tf
from util.utils import *
def init_w(w_shape,w_stddev,dtype):
    w = tf.Variable(
        tf.truncated_normal(
            w_shape
            , stddev=w_stddev
            , dtype=dtype
        )
        , dtype=dtype
    )

    return w

def init_b(b_shape,b_const,dtype):
    b = tf.Variable(
        tf.constant(
            b_const
            , shape=b_shape
            , dtype=dtype
        )
        , dtype=dtype
    )
    return b

def linear(input,w_shape,b_shape,test_input=None,b_const=0.1,w_stddev=0.01,dtype=tf.float32,activate = 'sigmoid'):
    w = init_w(w_shape,w_stddev,dtype)
    b = init_b(b_shape,b_const,dtype)

    output = tf.matmul(input,w)+b
    if test_input is not None:
        test_output = tf.matmul(test_input, w) + b
        if activate == None:
            return output,test_output
        elif activate == 'sigmoid':
            return tf.nn.sigmoid(output),tf.nn.sigmoid(test_output)
        elif activate == 'relu':
            return tf.nn.relu(output),tf.nn.relu(test_output)
        else:
            raise ('activate function error!!!')
    else:
        if activate == None:
            return output
        elif activate == 'sigmoid':
            return tf.nn.sigmoid(output)
        elif activate == 'relu':
            return tf.nn.relu(output)
        else:
            raise ('activate function error!!!')


def max2dpooling(input,kernel_size,strides,padding,test_input=None):
    if test_input is not None:
        return tf.nn.max_pool(input, kernel_size, strides, padding),tf.nn.max_pool(test_input, kernel_size, strides, padding)
    else:
        return tf.nn.max_pool(input,kernel_size,strides,padding)

def conv2d(
        input,
        filter_size,
        b_size,
        strides,
        padding,
        dtype=tf.float32,
        test_input=None,
        w_stddev=0.01,
        b_const = 0.1,
        activate = 'sigmoid'
    ):
    assert isinstance(filter_size,list) and isinstance(b_size,list)
    w = init_w(filter_size,w_stddev,dtype)
    b = init_b(b_size,b_const,dtype)
    # w = weight_variable(filter_size)
    # b = bias_variable(b_size)
    conv_output = tf.nn.conv2d(
        input,
        w,
        strides,
        padding=padding
    )
    if test_input is not None:
        test_conv_output = tf.nn.conv2d(
            test_input,
            w,
            strides,
            padding=padding
        )
        if activate == 'sigmoid':
            return tf.nn.sigmoid(conv_output + b),tf.nn.sigmoid(test_conv_output + b)
        elif activate == 'relu':
            return tf.nn.relu(conv_output + b),tf.nn.relu(test_conv_output + b)
        elif activate == 'elu':
            return tf.nn.elu(conv_output + b), tf.nn.elu(test_conv_output + b)
        elif activate == 'tanh':
            return tf.nn.tanh(conv_output + b),tf.nn.tanh(test_conv_output + b)
        elif activate == 'leaky_relu':
            return tf.nn.relu6(conv_output + b),tf.nn.relu6(test_conv_output + b)
            # return tf.nn.leaky_relu(conv_output + b)
        else:
            raise ('activate function error!!!')
    else:
        if activate == 'sigmoid':
            return tf.nn.sigmoid(conv_output + b)
        elif activate == 'relu':
            return tf.nn.relu(conv_output + b)
        elif activate == 'elu':
            return tf.nn.elu(conv_output + b)
        elif activate == 'tanh':
            return tf.nn.tanh(conv_output + b)
        elif activate == 'leaky_relu':
            return tf.nn.relu6(conv_output + b)
            # return tf.nn.leaky_relu(conv_output + b)
        else:
            raise ('activate function error!!!')
