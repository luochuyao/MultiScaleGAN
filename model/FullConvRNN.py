import tensorflow as tf
import keras.backend as K

def weight_variable(name,shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype= tf.float32)

def bias_variable(name,shape):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype= tf.float32)



class Conv2DRNNCell(object):

    def __init__(self, layer_param):
        self.input_channels = layer_param['input_channels']
        self.output_channels = layer_param['output_channels']
        self.input_to_state_kernel_size = layer_param['input_to_state_kernel_size']
        self.state_to_state_kernel_size = layer_param['state_to_state_kernel_size']

    def init_parameter(self, parameter_shape, initial_method='uniform'):
        if initial_method == 'uniform':
            parameter = tf.Variable(
                tf.random_uniform(
                    parameter_shape,
                    stddev=0.01
                )
            )
        elif initial_method == 'zero':
            parameter = tf.Variable(
                tf.constant(
                    value=0,
                    shape=parameter_shape,
                    dtype=tf.float32
                )
            )

        return parameter


class Conv2DGRUCell(Conv2DRNNCell):
    def __init__(self, layer_param):
        super(Conv2DGRUCell, self).__init__(layer_param)
        self.build()
    def build(self):
        input_gate_input_to_state_weight_shape = [
            self.input_to_state_kernel_size[0],
            self.input_to_state_kernel_size[1],
            self.input_channels,
            self.output_channels
        ]
        input_gate_biase_shape = [
            1,1,1,self.output_channels
        ]
        input_gate_state_to_state_weight_shape = [
            self.state_to_state_kernel_size[0],
            self.state_to_state_kernel_size[1],
            self.output_channels,
            self.output_channels
        ]

        forget_gate_input_to_state_weight_shape = [
            self.input_to_state_kernel_size[0],
            self.input_to_state_kernel_size[1],
            self.input_channels,
            self.output_channels
        ]
        forget_gate_biase_shape = [
            1, 1, 1, self.output_channels
        ]
        forget_gate_state_to_state_weight_shape = [
            self.state_to_state_kernel_size[0],
            self.state_to_state_kernel_size[1],
            self.output_channels,
            self.output_channels
        ]

        h_t_input_to_state_weight_shape = [
            self.input_to_state_kernel_size[0],
            self.input_to_state_kernel_size[1],
            self.input_channels,
            self.output_channels
        ]
        h_t_biase_shape = [
            1,1,1,self.output_channels
        ]
        h_t_state_to_state_weight_shape = [
            self.state_to_state_kernel_size[0],
            self.state_to_state_kernel_size[1],
            self.output_channels,
            self.output_channels
        ]

        self.input_gate_input_to_state_weight = weight_variable('input_gate_input_to_state_weight',input_gate_input_to_state_weight_shape)
        self.input_gate_state_to_state_weight = weight_variable('input_gate_state_to_state_weight',input_gate_state_to_state_weight_shape)
        self.input_gate_biase = bias_variable('input_gate_biase',input_gate_biase_shape)

        self.forget_gate_input_to_state_weight = weight_variable('forget_gate_input_to_state_weight',forget_gate_input_to_state_weight_shape)
        self.forget_gate_state_to_state_weight = weight_variable('forget_gate_state_to_state_weight',forget_gate_state_to_state_weight_shape)
        self.forget_gate_biase = bias_variable('forget_gate_biase',forget_gate_biase_shape)

        self.h_t_input_to_state_weight = weight_variable('h_t_input_to_state_weight',h_t_input_to_state_weight_shape)
        self.h_t_state_to_state_weight = weight_variable('h_t_state_to_state_weight',h_t_state_to_state_weight_shape)
        self.h_t_biase = bias_variable('h_t_biase',h_t_biase_shape)

    def __call__(self, input, hidden):
        h_tm1 = hidden
        input_gate = tf.nn.sigmoid(
            tf.nn.conv2d(
                input = input,
                filter=self.input_gate_input_to_state_weight,
                strides=(1, 1, 1, 1),
                padding='SAME',
            )
            + tf.nn.conv2d(
                input = h_tm1,
                filter = self.input_gate_state_to_state_weight,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + self.input_gate_biase
        )

        forget_gate = tf.nn.sigmoid(
            tf.nn.conv2d(
                input=input,
                filter = self.forget_gate_input_to_state_weight,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + tf.nn.conv2d(
                input = h_tm1,
                filter = self.forget_gate_state_to_state_weight,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + self.forget_gate_biase
        )

        h_t_ = tf.nn.tanh(
            tf.nn.conv2d(
                input = input,
                filter = self.h_t_input_to_state_weight,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + forget_gate * tf.nn.conv2d(
                input = h_tm1,
                filter = self.h_t_state_to_state_weight,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + self.h_t_biase
        )

        h_t = (1 - input_gate) * h_t_ + input_gate * h_tm1

        return h_t, h_t


class Conv2DLSTMCell(Conv2DRNNCell):

    def __init__(self, layer_param):
        super(Conv2DLSTMCell, self).__init__(layer_param)
        self.build()

    def build(self):
        # initial the weight from state to state
        w_h_shape = [
            self.state_to_state_kernel_size[0],
            self.state_to_state_kernel_size[1],
            self.output_channels,
            self.output_channels,
        ]
        w_hs = ['w_hi','w_hf','w_hc','w_ho']
        [self.w_hi, self.w_hf, self.w_hc, self.w_ho] = [weight_variable(w_hs[i], w_h_shape) for i in range(4)]

        # initial the weight from input to state
        w_i_shape = [
            self.input_to_state_kernel_size[0],
            self.input_to_state_kernel_size[1],
            self.input_channels,
            self.output_channels,
                     ]
        w_is = ['w_xi','w_xf','w_xc','w_xo']
        [self.w_xi, self.w_xf, self.w_xc, self.w_xo] = [weight_variable(w_is[i],w_i_shape) for i in range(4)]

        # initial the biase
        b_shape = (1,1,1,self.output_channels)
        b_s = ['b_i','b_f','b_c','b_o']
        [self.b_i, self.b_f, self.b_c, self.b_o] = [bias_variable(b_s[i],b_shape) for i in range(4)]

        # initial the weight of cell

        w_c_shape = (1, 1, 1, self.output_channels)
        [self.w_ci, self.w_cf, self.w_co] = [self.init_parameter(w_c_shape, initial_method='zero') for _ in range(3)]

    def __call__(self, input, hidden):
        h_tm1, c_tm1 = hidden

        input_gate = tf.nn.sigmoid(
            tf.nn.conv2d(
                input = input,
                filter = self.w_xi,
                strides = (1,1,1,1),
                padding = 'SAME',
            )
            + tf.nn.conv2d(
                input = h_tm1,
                filter = self.w_hi,
                strides = (1, 1, 1, 1),
                padding = 'SAME',
            )
            + self.b_i
            + tf.multiply(c_tm1 , self.w_ci)
        )

        forget_gate = tf.nn.sigmoid(
            tf.nn.conv2d(
                input = input,
                filter = self.w_xf,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + tf.nn.conv2d(
                input = h_tm1,
                filter = self.w_hf,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            + self.b_f
            + tf.multiply(c_tm1 , self.w_cf)
        )

        c_t = forget_gate * c_tm1 + input_gate * tf.nn.tanh(
            tf.nn.conv2d(
                input = input,
                filter = self.w_xc,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            +tf.nn.conv2d(
                input = h_tm1,
                filter = self.w_hc,
                strides = (1, 1, 1, 1),
                padding = 'SAME'
            )
            +self.b_c
        )

        output_gate = tf.nn.sigmoid(
            tf.nn.conv2d(
                input = input,
                padding = 'SAME',
                filter = self.w_xo,
                strides = (1, 1, 1, 1)
            )
            +tf.nn.conv2d(
                input = h_tm1,
                padding = 'SAME',
                filter = self.w_ho,
                strides = (1, 1, 1, 1)
            )
            +self.b_o
           + tf.multiply(c_t , self.w_co)
        )

        h_t = output_gate * tf.nn.tanh(c_t)

        return h_t, (h_t, c_t)


class Conv2DRNN(object):

    def __init__(self, name, cell_param, return_state=False, return_sequence=False):
        self.name = name
        self.cell_param = cell_param
        self.return_state = return_state
        self.return_sequence = return_sequence
        self.cell = None

    def init_parameter(self, parameter_shape, initial_method):
        if initial_method == 'uniform':
            parameter = tf.Variable(
                tf.truncated_normal(
                    parameter_shape,
                    stddev=0.01
                ),
                dtype=tf.float32
            )
        elif initial_method == 'zero':
            parameter = tf.Variable(
                tf.constant(
                    value=0,
                    shape=parameter_shape,
                    dtype=tf.float32
                ),
                dtype=tf.float32
            )
        return parameter

    def init_hidden(self, hidden_shape):
        pass

    def __call__(self, input, initial_state=None, reuse=False):

            if isinstance(self.cell, type(None)):
                raise ('the type of cell should not be None')
            else:
                pass
            input_shape = input.get_shape().as_list()
            time_step = input_shape[1]
            if initial_state == None:

                hidden_shape = (input_shape[0], input_shape[2], input_shape[3], self.cell_param['output_channels'])
                state = self.init_hidden(hidden_shape)
            else:
                # print('init state is not None')
                state = initial_state
            outputs = []

            for i in range(time_step):
                x_t = input[:, i, :, :, :]

                h, state = self.cell(x_t, state)

                outputs.append(h)

            outputs = tf.stack(outputs, axis=1)
            if self.return_state == True:
                if self.return_sequence == True:
                    return outputs, state
                else:
                    return state

            else:
                if self.return_sequence == True:

                    return outputs
                else:
                    return h


class Conv2DLSTM(Conv2DRNN):

    def __init__(self, name, cell_param, return_state=False, return_sequence=False):
        super(Conv2DLSTM, self).__init__(name, cell_param, return_state, return_sequence)
        with tf.variable_scope(self.name) as vs:
            print('name is:', self.name)
            self.cell = Conv2DLSTMCell(cell_param)

    def init_hidden(self, hidden_shape):
        h, c = self.init_parameter(hidden_shape, initial_method='zero'), self.init_parameter(hidden_shape,
                                                                                             initial_method='zero')
        return (h, c)


class Conv2DGRU(Conv2DRNN):

    def __init__(self, name, cell_param, return_state=False, return_sequence=False):
        super(Conv2DGRU, self).__init__(name, cell_param, return_state, return_sequence)

        with tf.variable_scope(self.name) as vs:
            print('name is:',self.name)
            self.cell = Conv2DGRUCell(cell_param)

    def init_hidden(self, hidden_shape):
        h = self.init_parameter(hidden_shape, initial_method='zero')
        return h