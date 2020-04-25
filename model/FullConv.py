import tensorflow as tf

def weight_variable(name,shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype= tf.float32)

def bias_variable(name,shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype= tf.float32)

class Conv(object):

    def __init__(self,name,layer_param):
        self.name = name
        self.input_channel = layer_param['input_channel']
        self.output_channel = layer_param['output_channel']
        self.kernel_size = layer_param['kernel_size']
        self.padding = layer_param['padding']
        self.activate = layer_param['activate']


    def init_parameter(self,parameter_shape,initial_method):

        if initial_method=='uniform':
            parameter = tf.Variable(
                tf.truncated_normal(
                    parameter_shape,
                    stddev=0.01
                )
            )
        elif initial_method == 'zero':
            parameter = tf.Variable(
                tf.constant(
                    value=0,
                    shape=parameter_shape
                )
            )
        return parameter

    def build(self):
        pass

class Pooling(object):
    def __init__(self,layer_param):
        self.pool_mode = layer_param['pool_mode']
        self.pool_size = layer_param['pool_size']
        self.strides = layer_param['strides']
        self.padding = layer_param['padding']


    def __call__(self, input):
        if self.pool_mode == 'avg':
            return tf.nn.avg_pool(
                value = input,
                ksize = self.pool_size,
                strides = self.strides,
                padding = self.padding
            )
        elif self.pool_mode == 'max':
            return tf.nn.max_pool(
                value = input,
                ksize = self.pool_size,
                strides = self.strides,
                padding = self.padding
            )
        else:
            raise ('the mode of pooling is error')

class Conv2D(Conv):

    def __init__(self,name,layer_param):
        super(Conv2D, self).__init__(name,layer_param)
        self.build()

    def build(self):
        with tf.variable_scope(self.name) as vs:
            filter_shape = [
                self.kernel_size[0],
                self.kernel_size[1],
                self.input_channel,
                self.output_channel,
            ]
            bias_shape = [
                1,1,1,self.output_channel
            ]
            self.filter = weight_variable('w',filter_shape)
            self.bias = bias_variable('b',bias_shape)

    def __call__(self,input):

        conv2d = tf.nn.conv2d(
            input = input,
            filter = self.filter,
            strides = [1,1,1,1],
            padding = self.padding
        ) + self.bias

        if self.activate == None:
            return conv2d
        elif self.activate == 'sigmoid':
            return tf.nn.sigmoid(conv2d)
        elif self.activate == 'relu':
            return tf.nn.relu(conv2d)
        elif self.activate == 'tanh':
            return tf.nn.tanh(conv2d)
