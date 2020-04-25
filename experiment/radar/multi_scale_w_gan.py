
import sys
import os
import datetime

# runing the progress by nohup
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])

from model.WDiscriminator import *
from loss.loss_functions import *
import yaml
import math
from util import ops
from util.utils import *
from datetime import datetime
from data.radar_sequence_iterator import SequenceRadarDataIterator


class MultiScaleWGAN(object):
    def __init__(self,
                 name,
                 train_iter,
                 test_iter,
                 valid_iter,
                 scale_kernel_sizes,
                 scale_channels,
                 d_net,
                 info,
                 epoches = 10000,
                 train_batch_size = 8,
                 test_batch_size = 8,
                 ):
        self.name = name
        self.gen_name = name+'_generaotr'
        self.epoches = epoches
        self.info = info
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_channels = scale_channels
        self.d_net = d_net
        self.num_scale_nets = len(self.scale_channels)
        self.train_data_iter = train_iter
        self.test_data_iter = test_iter
        self.valid_data_iter = valid_iter
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.test_step = self.info['TRAINING']['TEST_STEP']
        self.train_width = info['TRAINING']['WIDTH_TRAIN']
        self.train_height = info['TRAINING']['HEIGHT_TRAIN']
        self.test_width = info['TESTING']['WIDTH_TEST']
        self.test_height = info['TESTING']['HEIGHT_TEST']
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True


        self.train_input_frame = tf.placeholder(
            tf.float32,
            [
                None,
                self.train_height,
                self.train_width,
                4
            ]
        )
        self.train_target_frame = tf.placeholder(
            tf.float32,
            [
                None,
                self.train_height,
                self.train_width,
                1
            ]
        )
        self.test_input_frame = tf.placeholder(
            tf.float32,
            [
                None,
                self.test_height,
                self.test_width,
                4
            ]
        )
        self.test_target_frame = tf.placeholder(
            tf.float32,
            [
                None,
                self.test_height,
                self.test_width,
                1
            ]
        )
        self.scale_preds,\
        self.scale_targets,\
        self.test_scale_preds, \
        self.test_scale_targets = self.build_model(
            self.train_input_frame,
            self.train_target_frame,
            self.test_input_frame,
            self.test_target_frame
        )

        self.d_output = self.d_net(self.scale_preds,self.scale_targets)

        self.d_real = []
        self.d_fake = []

        for i,d_o in enumerate(self.d_output):
            self.d_fake.append(d_o[:self.scale_targets[i].get_shape().as_list()[0]])
            self.d_real.append(d_o[self.scale_targets[i].get_shape().as_list()[0]:])

        self.loss = combined_loss(
            self.scale_preds, self.scale_targets, info
        )

        self.g_loss = tf.reduce_mean(self.d_fake)
        self.d_loss = tf.reduce_mean(self.d_real) - tf.reduce_mean(self.d_fake)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        d_learning_rate = tf.train.exponential_decay(
            learning_rate=self.info['MODEL_PARAMETER_D']['LEARNING_RATE_D'],
            global_step=self.global_step,
            decay_steps=3000,
            decay_rate=0.8,
            staircase=True
        )
        g_learning_rate = tf.train.exponential_decay(
            learning_rate=self.info['MODEL_PARAMETER_G']['LEARNING_RATE_G'],
            global_step=self.global_step,
            decay_steps=3000,
            decay_rate=0.8,
            staircase=True
        )


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rms= tf.train.RMSPropOptimizer(learning_rate=d_learning_rate) \
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_rms = tf.train.RMSPropOptimizer(learning_rate=g_learning_rate) \
                .minimize(self.g_loss, var_list=self.vars)
            self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=g_learning_rate) \
                .minimize(self.loss, var_list=self.vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars if 'RMSProp' not in v.name]
        self.sess = tf.Session(config=tfconfig)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        print('initalize finnish')



    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.gen_name in var.name]

    def build_model(self,
                    train_input_frames,
                    train_target_frames,
                    test_input_frames,
                    test_target_frames,
                    ):
        self.scale_channels = self.info['MODEL_PARAMETER_G']['SCALE_CHANNELS']
        self.scale_kernel_sizes = self.info['MODEL_PARAMETER_G']['SCALE_KERNEL_SIZES']
        self.num_scale_nets = len(self.scale_channels)
        with tf.variable_scope(self.gen_name):
            with tf.variable_scope(self.gen_name + '_data'):
                train_height = train_input_frames.get_shape().as_list()[1]
                train_width = train_target_frames.get_shape().as_list()[2]
                test_height = test_input_frames.get_shape().as_list()[1]
                test_width = test_target_frames.get_shape().as_list()[2]
            train_scale_preds = []
            train_scale_targets = []
            test_scale_preds = []
            test_scale_targets = []
            for scale_num in range(self.num_scale_nets):
                with tf.variable_scope(self.gen_name + '_scale' + str(scale_num)):
                    with tf.variable_scope(self.gen_name + '_scale' + str(scale_num) + 'convolution'):
                        scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                        scale_train_height = int(train_height * scale_factor)
                        scale_train_width = int(train_width * scale_factor)
                        scale_test_height = int(test_height * scale_factor)
                        scale_test_width = int(test_width * scale_factor)
                        scale_train_input = tf.image.resize_images(train_input_frames,
                                                                   [scale_train_height, scale_train_width])
                        scale_test_input = tf.image.resize_images(test_input_frames,
                                                                  [scale_test_height, scale_test_width])
                        scale_train_target = tf.image.resize_images(train_target_frames,
                                                                    [scale_train_height, scale_train_width])
                        scale_test_target = tf.image.resize_images(test_target_frames,
                                                                   [scale_test_height, scale_test_width])

                        if scale_num > 0:
                            last_scale_train_pred = train_scale_preds[-1]
                            last_scale_test_pred = test_scale_preds[-1]
                            last_gen_train_frames = tf.image.resize_images(last_scale_train_pred,
                                                                           [scale_train_height, scale_train_width])
                            last_gen_test_frames = tf.image.resize_images(last_scale_test_pred,
                                                                          [scale_test_height, scale_test_width])

                            scale_train_input = tf.concat([scale_train_input, last_gen_train_frames], 3)
                            scale_test_input = tf.concat([scale_test_input, last_gen_test_frames], 3)
                        else:
                            last_scale_train_pred = None
                            last_scale_test_pred = None

                        for i in range(len(self.scale_kernel_sizes[scale_num])):
                            if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                                scale_train_input, scale_test_input = ops.conv2d(
                                    input=scale_train_input,
                                    test_input=scale_test_input,
                                    filter_size=[
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_channels[scale_num][i],
                                        self.scale_channels[scale_num][i + 1]
                                    ],
                                    b_size=[self.scale_channels[scale_num][i + 1]],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    dtype=tf.float32,
                                    activate='tanh'
                                )
                            else:

                                scale_train_input, scale_test_input = ops.conv2d(
                                    input=scale_train_input,
                                    test_input=scale_test_input,
                                    filter_size=[
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_kernel_sizes[scale_num][i],
                                        self.scale_channels[scale_num][i],
                                        self.scale_channels[scale_num][i + 1]
                                    ],
                                    b_size=[self.scale_channels[scale_num][i + 1]],
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    dtype=tf.float32,
                                    activate='leaky_relu'
                                )

                        scale_train_pred = scale_train_input
                        scale_test_pred = scale_test_input

                        train_scale_preds.append(scale_train_pred)
                        test_scale_preds.append(scale_test_pred)
                        train_scale_targets.append(scale_train_target)
                        test_scale_targets.append(scale_test_target)

            return train_scale_preds, train_scale_targets, test_scale_preds, test_scale_targets

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print('initalize finnish')
        print('start training!!!')
        tolerate_iter = 0
        best_hss = []
        best_csi = []
        best_hss_avg = 0
        best_csi_avg = 0
        for step in range(self.epoches):

            frame_seq = self.train_data_iter.sample(batch_size=self.train_batch_size)

            frame_seq[frame_seq > 80.0] = 0.0
            frame_seq[frame_seq < 15.0] = 0.0
            frame_seq = normalization(frame_seq)
            train_x = frame_seq[:, :, :, :4].astype(np.float32)
            train_y = frame_seq[:, :, :, -1:].astype(np.float32)
            train_input_frames = train_x
            train_target_frames = train_y

            # train discriminator
            self.sess.run(self.d_clip)
            self.sess.run(self.d_rms,feed_dict={
                self.train_input_frame: train_input_frames,
                self.train_target_frame: train_target_frames
            })

            #train generator
            self.sess.run(self.g_rms, feed_dict={
                self.train_input_frame: train_input_frames,
                self.train_target_frame: train_target_frames
            })

            for _ in range(4):
                self.sess.run(
                    self.adam_optimizer,
                    feed_dict={
                        self.train_input_frame: train_input_frames,
                        self.train_target_frame: train_target_frames
                    }
                )

                # print the training information
                if (step + 1) % self.info['TRAINING']['DISTPLAY_STEP'] == 0:
                    loss = self.sess.run(
                        self.loss,
                        feed_dict={
                            self.train_input_frame: train_input_frames,
                            self.train_target_frame: train_target_frames
                        }
                    )
                    print('*' * 50)
                    print('step is :', str(step + 1))
                    print('train batch loss is:', str(loss))
                    print('*' * 50)
                    break

            if (step+1)%self.test_step==0:
                HSS, CSI, avg_HSS, avg_CSI = self.test()
                print('the HSS is:', HSS, avg_HSS)
                print('the CSI is:', CSI, avg_CSI)
                if avg_HSS>best_hss_avg or avg_CSI>best_csi_avg:
                    best_hss_avg = avg_HSS
                    best_csi_avg = avg_CSI
                    best_hss = HSS
                    best_csi = CSI
                    tolerate_iter = 0
                    self.save_model()
                else:
                    tolerate_iter += 1
                    if tolerate_iter==3:
                        print('the best HSS is:', best_hss, best_hss_avg)
                        print('the best CSI is:', best_csi, best_csi_avg)
                        break

    # saving model
    def save_model(self, model_name=None):

        model_fold = os.path.join(self.base_path, self.info['MODEL_SAVE_DIR'])

        if not os.path.exists(model_fold):
            os.mkdir(model_fold)
        else:
            pass
        if model_name == None:
            self.saver.save(
                self.sess,
                os.path.join(model_fold, 'radar_model.ckpt')
            )
        else:
            self.saver.save(
                self.sess,
                os.path.join(model_fold, model_name)
            )
        print('model saved')

    # loading model
    def load_model(self, model_name=None):
        model_fold = os.path.join(self.base_path, self.info['MODEL_SAVE_DIR'])
        if not os.path.exists(model_fold):
            raise ('the path of model is null')
        else:
            if model_name == None:
                self.saver.restore(
                    self.sess,
                    os.path.join(model_fold, 'radar_model.ckpt')
                )
            else:
                self.saver.restore(
                    self.sess,
                    os.path.join(model_fold, model_name)
                )
        print('model has been loaded')

    # test the model in terms of MSE
    def test(self):
        count = 0
        MSE = 0
        for i in range(self.test_data_iter.number_sample):
            frame_dat, names, file_name, = self.test_data_iter.sample_process(i)
            frame_dat[frame_dat > 80] = 0
            frame_dat[frame_dat < 15] = 0
            frame_dat = normalization(frame_dat)
            for j in range((len(frame_dat) - 13) // 5):
                print('current process ' + str(i) + '/' + str(
                    self.test_data_iter.number_sample) + '\tfinish ' + str(
                    100 * j / (len(frame_dat) - 13)) + '%')
                current_process = frame_dat[j * 5:j * 5 + 14]

                test_input_frame = current_process[:4]
                test_input_frame = test_input_frame.transpose((1, 2, 0))[np.newaxis, :, :, :, ]
                tars = current_process[-10:]
                img_out = []
                for img_index in range(len(current_process) - 4):
                    test_output1 = self.sess.run(
                        self.test_scale_preds[-1],
                        feed_dict={
                            self.test_input_frame: test_input_frame[:, :, 0:450, :],
                        }
                    )
                    test_output2 = self.sess.run(
                        self.test_scale_preds[-1],
                        feed_dict={
                            self.test_input_frame: test_input_frame[:, :, 450:, :],
                        }
                    )
                    test_output = np.concatenate((test_output1, test_output2), axis=2)

                    img_out.append(test_output[0, :, :, 0])
                    test_input_frame = np.concatenate([test_input_frame[:, :, :, -3:], test_output], axis=3)

                img_out = np.stack(img_out, 0)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1
        MSE = MSE / count
        return MSE

    # valid the model in terms of MSE
    def valid(self):
        count = 0
        MSE = 0
        for i in range(self.valid_data_iter.number_sample):
            frame_dat, names, file_name, = self.valid_data_iter.sample_process(i)
            frame_dat[frame_dat > 80] = 0
            frame_dat[frame_dat < 15] = 0
            frame_dat = normalization(frame_dat)
            for j in range((len(frame_dat) - 13) // 5):
                print('current process ' + str(i) + '/' + str(
                    self.valid_data_iter.number_sample) + '\tfinish ' + str(
                    100 * j / (len(frame_dat) - 13)) + '%')
                current_process = frame_dat[j * 5:j * 5 + 14]

                test_input_frame = current_process[:4]
                test_input_frame = test_input_frame.transpose((1, 2, 0))[np.newaxis, :, :, :, ]
                tars = current_process[-10:]
                img_out = []
                for img_index in range(len(current_process) - 4):
                    test_output1 = self.sess.run(
                        self.test_scale_preds[-1],
                        feed_dict={
                            self.test_input_frame: test_input_frame[:, :, 0:450, :],
                        }
                    )
                    test_output2 = self.sess.run(
                        self.test_scale_preds[-1],
                        feed_dict={
                            self.test_input_frame: test_input_frame[:, :, 450:, :],
                        }
                    )
                    test_output = np.concatenate((test_output1, test_output2), axis=2)

                    img_out.append(test_output[0, :, :, 0])
                    test_input_frame = np.concatenate([test_input_frame[:, :, :, -3:], test_output], axis=3)

                img_out = np.stack(img_out, 0)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1
        MSE = MSE / count
        return MSE


def main():
    import sys

    model_name = 'Multi_Scale_WGAN'

    base_path = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(base_path, 'config', model_name + '.yml')
    configuration = yaml.load(open(config_path))


    d_net = WDiscriminatorModel(
        name=configuration['NAME']+'_discriminator',
        info=configuration,
        scale_kernel_sizes=configuration['MODEL_PARAMETER_D']['SCALE_KERNEL_SIZES'],
        scale_channels=configuration['MODEL_PARAMETER_D']['SCALE_CHANNELS'],
        scale_fully_connect_layer_size=configuration['MODEL_PARAMETER_D']['SCALE_FULLY_CONNECT_LAYER_SIZES'],
        height_train=configuration['TRAINING']['HEIGHT_TRAIN'],
        width_train=configuration['TRAINING']['WIDTH_TRAIN'],
    )
    print('discriminator create successful')

    # initialize the data iterator
    train_data_itertor = \
        SequenceRadarDataIterator(
            root_path=configuration['TRAIN_DATA_SAVE_DIR'],
            mode='Train',
            clip_height=configuration['TRAINING']['HEIGHT_TRAIN'],
            clip_width=configuration['TRAINING']['WIDTH_TRAIN'],
        )

    test_data_itertor = \
        SequenceRadarDataIterator(
            root_path=configuration['TEST_DATA_SAVE_DIR'],
            mode='Test',
        )

    valid_data_iterator = \
        SequenceRadarDataIterator(
            root_path=configuration['TRAIN_DATA_SAVE_DIR'],
            mode='Valid',
        )

    wgan = MultiScaleWGAN(
        name=configuration['NAME'],
        scale_channels=configuration['MODEL_PARAMETER_G']['SCALE_CHANNELS'],
        scale_kernel_sizes=configuration['MODEL_PARAMETER_G']['SCALE_KERNEL_SIZES'],
        train_iter=train_data_itertor,
        test_iter=test_data_itertor,
        valid_iter=valid_data_iterator,
        train_batch_size=configuration['TRAINING']['BATCH_SIZE'],
        test_batch_size=configuration['TESTING']['BATCH_SIZE'],
        d_net=d_net,
        info=configuration,
    )
    wgan.save_model()
    wgan.load_model()
    wgan.valid()




if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        main()
