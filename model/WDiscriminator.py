import tensorflow as tf
from util.ops import *

def conv_out_size( i, p, k, s):
    if p == 'SAME':
        p = k // 2
    elif p == 'VALID':
        p = 0
    else:
        raise ValueError('p must be "SAME" or "VALID".')

    return int(((i + (2 * p) - k) / s) + 1)


class WDiscriminatorModel(object):
    def __init__(self,
                 name,
                 info,
                 scale_kernel_sizes,
                 scale_channels,
                 scale_fully_connect_layer_size,
                 height_train,
                 width_train):
        self.info = info
        self.name = name
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_channels = scale_channels
        self.scale_fully_connect_layer_sizes = scale_fully_connect_layer_size
        self.height_train = height_train
        self.width_train = width_train
        self.num_scale_nets = len(scale_kernel_sizes)
        # self.define_parameter()
        # print(self.name,' initialization finish ')

    def define_parameter(self):
        self.scale_conv_ws = []
        self.scale_conv_bs = []
        self.scale_fc_ws = []
        self.scale_fc_bs = []
        with tf.variable_scope(self.name) as vs:
            for scale_num in range(self.num_scale_nets):
                with tf.variable_scope(self.name + '_scale_net_parameter' + str(scale_num)):
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                    scale_height = int(self.height_train * scale_factor)
                    scale_width = int(self.width_train * scale_factor)
                    channels = self.scale_channels[scale_num]
                    kernel_sizes = self.scale_kernel_sizes[scale_num]
                    fully_connect_sizes = self.scale_fully_connect_layer_sizes[scale_num]

                    conv_ws = []
                    conv_bs = []
                    last_out_height = scale_height
                    last_out_width = scale_width

                    for i in range(len(kernel_sizes)):
                        conv_ws.append(
                            tf.Variable(
                                tf.truncated_normal(
                                    [
                                        kernel_sizes[i],
                                        kernel_sizes[i],
                                        channels[i],
                                        channels[i + 1]
                                    ]
                                    , stddev=self.info['UTIL_PARAMETER']['W_STDDEV']
                                    , dtype=tf.float32
                                ),
                                dtype=tf.float32
                            )
                        )
                        conv_bs.append(
                            tf.Variable(
                                tf.constant(
                                    self.info['UTIL_PARAMETER']['B_CONST']
                                    , shape=[channels[i + 1]]
                                    , dtype=tf.float32,
                                )
                                , dtype=tf.float32
                            )
                        )

                        last_out_height = conv_out_size(
                            last_out_height, self.info['MODEL_PARAMETER_D']['PADDING_D'], kernel_sizes[i], 1)
                        last_out_width = conv_out_size(
                            last_out_width, self.info['MODEL_PARAMETER_D']['PADDING_D'], kernel_sizes[i], 1)

                    fully_connect_sizes.insert(
                        0, int((last_out_height / 2) * (last_out_width / 2) * channels[-1]))

                    self.scale_conv_ws.append(conv_ws)
                    self.scale_conv_bs.append(conv_bs)
                    fc_ws = []
                    fc_bs = []

                    for i in range(len(fully_connect_sizes) - 1):
                        fc_ws.append(
                            tf.Variable(
                                tf.truncated_normal(
                                    [
                                        int(fully_connect_sizes[i]),
                                        int(fully_connect_sizes[i + 1])
                                    ]
                                    , stddev=self.info['UTIL_PARAMETER']['W_STDDEV']
                                    , dtype=tf.float32
                                )
                                , dtype=tf.float32
                            )
                        )
                        fc_bs.append(
                            tf.Variable(
                                tf.constant(
                                    self.info['UTIL_PARAMETER']['B_CONST']
                                    , shape=[fully_connect_sizes[i + 1]]
                                    , dtype=tf.float32
                                )
                                , dtype=tf.float32
                            )
                        )
                    self.scale_fc_ws.append(fc_ws)
                    self.scale_fc_bs.append(fc_bs)


    def __call__(self, scale_pred_frames,scale_target_frames):
        scale_input_frames = []
        for i in range(len(scale_pred_frames)):
            scale_input_frames.append(
                tf.concat([scale_pred_frames[i],scale_target_frames[i]],axis=0)
            )

        with tf.variable_scope(self.name):

            scale_preds = []

            for scale_num in range(self.num_scale_nets):
                with tf.variable_scope(self.name + '_scale_net' + str(scale_num)):
                    input_frames = scale_input_frames[scale_num]
                    batch_size = tf.shape(input_frames)[0]
                    preds = tf.zeros([batch_size, 1])
                    last_input = input_frames
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                    scale_height = int(self.height_train * scale_factor)
                    scale_width = int(self.width_train * scale_factor)
                    last_out_height = scale_height
                    last_out_width = scale_width
                    for i in range(len(self.scale_kernel_sizes[scale_num])):
                        with tf.variable_scope(self.name + '_scale' + str(scale_num) + '_convolutions_'+str(i)):
                            last_input = conv2d(
                                input = last_input,
                                filter_size = [
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_channels[scale_num][i],
                                        self.scale_channels[scale_num][i + 1]
                                ],
                                b_size = [
                                    self.scale_channels[scale_num][i + 1]
                                ],
                                strides = [1,1,1,1],
                                padding = self.info['MODEL_PARAMETER_D']['PADDING_D'],
                                dtype = tf.float32,
                                activate = 'relu'
                            )
                        last_out_height = conv_out_size(
                            last_out_height,
                            self.info['MODEL_PARAMETER_D']['PADDING_D'],
                            self.scale_kernel_sizes[scale_num][i],
                            1
                        )
                        last_out_width = conv_out_size(
                            last_out_width,
                            self.info['MODEL_PARAMETER_D']['PADDING_D'],
                            self.scale_kernel_sizes[scale_num][i],
                            1
                        )
                    self.scale_fully_connect_layer_sizes[scale_num].insert(
                            0, int((last_out_height / 2) * (last_out_width / 2) * self.scale_channels[scale_num][-1]))
                    preds = last_input

                with tf.variable_scope(self.name + '_scale' + str(scale_num) + '_pooling'):
                    preds = max2dpooling(
                        input = preds,
                        kernel_size = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding=self.info['MODEL_PARAMETER_D']['PADDING_D'])

                    shape = preds.get_shape().as_list()

                    preds = tf.reshape(preds, [-1, shape[1] * shape[2] * shape[3]])

                with tf.variable_scope(self.name + '_scale' + str(scale_num) + '_fully-connected'):
                    for i in range(len(self.scale_fully_connect_layer_sizes[scale_num])-1):
                        if i == len(self.scale_fully_connect_layer_sizes[scale_num]) - 2:
                            preds = linear(
                                input=preds,
                                w_shape=[
                                    int(self.scale_fully_connect_layer_sizes[scale_num][i]),
                                    int(self.scale_fully_connect_layer_sizes[scale_num][i + 1])
                                ],
                                b_shape=[
                                    self.scale_fully_connect_layer_sizes[scale_num][i + 1]
                                ],
                                b_const=0.1,
                                dtype=tf.float32,
                                activate=None
                            )
                        else:
                            preds = linear(
                                input = preds,
                                w_shape =  [
                                        int(self.scale_fully_connect_layer_sizes[scale_num][i]),
                                        int(self.scale_fully_connect_layer_sizes[scale_num][i + 1])
                                    ],
                                b_shape = [
                                    self.scale_fully_connect_layer_sizes[scale_num][i + 1]
                                ],
                                b_const = 0.1,
                                dtype = tf.float32,
                                activate='relu'
                            )

                    preds = tf.clip_by_value(preds, 0.1, 0.9)

                    scale_preds.append(preds)

            return scale_preds

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]