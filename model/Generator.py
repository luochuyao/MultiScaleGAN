from loss.loss_functions import *
from util.ops import *
import tensorflow as tf


class GeneratorModel(object):

    def __init__(self,
                 name,
                 info,
                 scale_kernel_sizes,
                 scale_channels,
                ):
        self.name = name + '_generator'
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_channels = scale_channels
        self.num_scale_nets = len(self.scale_channels)
        self.info = info
        # self.define_parameter()
        # print(self.name, ' initialization finish ')

    def define_parameter(self):
        with tf.variable_scope(self.name):
            self.scale_ws = []
            self.scale_bs = []
            for scale_num in range(self.num_scale_nets):
                with tf.variable_scope(self.name + '_scale' + str(scale_num)):
                    with tf.variable_scope(self.name + '_scale' + str(scale_num) + 'parameter'):
                        ws = []
                        bs = []
                        for i in range(len(self.scale_kernel_sizes[scale_num])):
                            ws.append(
                                tf.Variable(
                                    tf.truncated_normal(
                                        [
                                            self.scale_kernel_sizes[scale_num][i],
                                            self.scale_kernel_sizes[scale_num][i],
                                            self.scale_channels[scale_num][i],
                                            self.scale_channels[scale_num][i + 1]
                                        ]
                                        ,stddev=self.info['UTIL_PARAMETER']['W_STDDEV']
                                        ,dtype=tf.float16
                                        )
                                    ,dtype=tf.float16
                                )
                            )
                            bs.append(
                                tf.Variable(
                                    tf.constant(
                                        self.info['UTIL_PARAMETER']['B_CONST']
                                        ,shape=
                                            [
                                                self.scale_channels[scale_num][i + 1]
                                            ]
                                        ,dtype=tf.float16
                                    )
                                    ,dtype=tf.float16
                                )
                            )
                        self.scale_ws.append(ws)
                        self.scale_bs.append(bs)


    # def __call__(self,input_frames,target_frames):
    #
    #     with tf.variable_scope(self.name):
    #         with tf.variable_scope(self.name + '_data'):
    #             height = input_frames.get_shape().as_list()[1]
    #             width = target_frames.get_shape().as_list()[2]
    #         scale_preds= []
    #         scale_targets = []
    #
    #         for scale_num in range(self.num_scale_nets):
    #             with tf.variable_scope(self.name + '_scale' + str(scale_num)):
    #                 with tf.variable_scope(self.name + '_scale' + str(scale_num) + 'convolution'):
    #                     scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
    #                     scale_height = int(height * scale_factor)
    #                     scale_width = int(width * scale_factor)
    #                     scale_input = tf.image.resize_images(input_frames,
    #                                                                [scale_height, scale_width])
    #                     scale_target = tf.image.resize_images(target_frames,
    #                                                                 [scale_height, scale_width])
    #
    #                     scale_input = tf.cast(scale_input,dtype=tf.float16)
    #                     scale_target = tf.cast(scale_target,dtype=tf.float16)
    #
    #                     if scale_num > 0:
    #                         last_scale_pred = scale_preds[scale_num - 1]
    #                         last_gen_frames = tf.image.resize_images(last_scale_pred,
    #                                                                        [scale_height, scale_width])
    #                         last_gen_frames = tf.cast(last_gen_frames,dtype=tf.float16)
    #
    #                         scale_input = tf.concat([scale_input, last_gen_frames], 3)
    #                     else:
    #                         last_scale_pred = None
    #
    #                     for i in range(len(self.scale_kernel_sizes[scale_num])):
    #                         scale_input = tf.nn.conv2d(
    #                             scale_input, self.scale_ws[scale_num][i], [1, 1, 1, 1],
    #                             padding=self.info['MODEL_PARAMETER_G']['PADDING_G']
    #                         )
    #                         if i == len(self.scale_kernel_sizes[scale_num]) - 1:
    #                             scale_input = tf.nn.sigmoid(scale_input + self.scale_bs[scale_num][i])
    #                         else:
    #                             scale_input = tf.nn.relu(scale_input + self.scale_bs[scale_num][i])
    #
    #                     scale_pred = scale_input
    #
    #                     scale_preds.append(scale_pred)
    #                     scale_targets.append(scale_target)
    #
    #         return scale_preds, scale_targets

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

    def __call__(self,input_frames,target_frames):

        with tf.variable_scope(self.name):
            with tf.variable_scope(self.name + '_data'):
                height = input_frames.get_shape().as_list()[1]
                width = target_frames.get_shape().as_list()[2]
            scale_preds= []
            scale_targets = []

            for scale_num in range(self.num_scale_nets):
                with tf.variable_scope(self.name + '_scale' + str(scale_num)):
                    with tf.variable_scope(self.name + '_scale' + str(scale_num) + '_convolution'):
                        scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                        scale_height = int(height * scale_factor)
                        scale_width = int(width * scale_factor)
                        scale_input = tf.image.resize_images(input_frames,
                                                                   [scale_height, scale_width])
                        scale_target = tf.image.resize_images(target_frames,
                                                                    [scale_height, scale_width])

                        scale_input = tf.cast(scale_input,dtype=tf.float16)
                        scale_target = tf.cast(scale_target,dtype=tf.float16)

                        if scale_num > 0:
                            last_scale_pred = scale_preds[-1]
                            last_gen_frames = tf.image.resize_images(last_scale_pred,
                                                                           [scale_height, scale_width])
                            last_gen_frames = tf.cast(last_gen_frames,dtype=tf.float16)

                            scale_input = tf.concat([scale_input, last_gen_frames], 3)
                        else:
                            last_scale_pred = None

                        for i in range(len(self.scale_kernel_sizes[scale_num])):

                            if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                                scale_input = conv2d(
                                    input=scale_input,
                                    filter_size=[
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_channels[scale_num][i],
                                        self.scale_channels[scale_num][i + 1]
                                    ],
                                    b_size=[self.scale_channels[scale_num][i + 1]],
                                    strides=[1, 1, 1, 1],
                                    padding=self.info['MODEL_PARAMETER_G']['PADDING_G'],
                                    dtype=tf.float16,
                                    activate='tanh'
                                )
                            else:
                                scale_input = conv2d(
                                    input=scale_input,
                                    filter_size=[
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_channels[scale_num][i],
                                        self.scale_channels[scale_num][i + 1]
                                    ],
                                    b_size=[self.scale_channels[scale_num][i + 1]],
                                    strides=[1, 1, 1, 1],
                                    padding=self.info['MODEL_PARAMETER_G']['PADDING_G'],
                                    dtype=tf.float16,
                                    activate='leaky_relu'
                                )

                        scale_pred = scale_input

                        scale_preds.append(scale_pred)
                        scale_targets.append(scale_target)

            return scale_preds, scale_targets


