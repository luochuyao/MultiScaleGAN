import sys
import os
import datetime

# runing the progress by nohup
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

from loss.loss_functions import *
import yaml
import math
from util import ops
from util.utils import *
from datetime import datetime
from scipy.misc import *
from data.radar_sequence_iterator import SequenceRadarDataIterator

def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

class MutilScaleCNN(object):

    def __init__(self,
                 name,
                 train_iter,
                 test_iter,
                 valid_iter,
                 info
                 ):
        self.gen_name = name
        self.train_data_iter = train_iter
        self.test_data_iter = test_iter
        self.valid_data_iter = valid_iter
        self.info = info
        self.test_step = self.info['TRAINING']['TEST_STEP']
        self.epoches = self.info['TRAINING']['EPOCHES']
        self.train_batch_size = self.info['TRAINING']['BATCH_SIZE']
        self.test_batch_size = self.info['TESTING']['BATCH_SIZE']
        self.train_width = info['TRAINING']['WIDTH_TRAIN']
        self.train_height = info['TRAINING']['HEIGHT_TRAIN']
        self.test_width = info['TESTING']['WIDTH_TEST']
        self.test_height = info['TESTING']['HEIGHT_TEST']
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        # create z
        self.g_para = {}

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
        self.scale_preds, \
        self.scale_targets,\
        self.test_scale_preds, \
        self.test_scale_targets = self.build_model(
            self.train_input_frame,
            self.train_target_frame,
            self.test_input_frame,
            self.test_target_frame
        )

        self.loss = combined_loss(self.scale_preds,self.scale_targets,self.info)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.info['MODEL_PARAMETER_G']['LEARNING_RATE_G']) \
                .minimize(self.loss, var_list=self.vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver(max_to_keep=1)

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
                                scale_train_input,scale_test_input = ops.conv2d(
                                    input=scale_train_input,
                                    test_input = scale_test_input,
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

            return train_scale_preds, train_scale_targets,test_scale_preds,test_scale_targets

    def train(self):

        tolerate_iter = 0
        best_loss = math.inf
        start_time = datetime.now()
        print('start training!!!')
        for step in range(self.epoches):
            frame_seq = self.train_data_iter.sample(batch_size=self.train_batch_size)
            frame_seq[frame_seq > 80.0] = 0.0
            frame_seq[frame_seq < 15.0] = 0.0
            frame_seq = normalization(frame_seq)
            train_x = frame_seq[:, :, :, :4].astype(np.float32)
            train_y = frame_seq[:, :, :, -1:].astype(np.float32)
            train_input_frames = train_x
            train_target_frames = train_y

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


            if (step+1)%self.test_step==0:
                valid_loss = self.valid()
                if valid_loss > best_loss:
                    best_loss = valid_loss
                    tolerate_iter = 0
                    self.save_model()
                else:
                    tolerate_iter += 1
                    if tolerate_iter == self.info['TRAINING']['LOSS_LIMIT']:
                        print('the best MSE is:', best_loss)
                        self.load_model()
                        self.test()
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
        model_fold = '/home/ices/PycharmProject/multi_scale_gan/model_lib/multi_scale_cnn_model_lib/radar/'
        # model_fold = os.path.join(self.base_path, self.info['MODEL_SAVE_DIR'])
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
                    test_output = self.sess.run(
                        self.test_scale_preds[-1],
                        feed_dict={
                            self.test_input_frame: test_input_frame,
                        }
                    )
                    img_out.append(test_output[0, :, :, 0])
                    test_input_frame = np.concatenate([test_input_frame[:, :, :, -3:], test_output], axis=3)

                img_out = np.stack(img_out, 0)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1


        MSE = MSE / count
        return MSE

    def read_files(self, path_list):
        imgs = []
        for path in path_list:
            img = imread(path)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs[imgs > 80] = 0
        imgs[imgs < 15] = 0
        imgs = normalization(imgs)
        imgs = imgs.transpose((1, 2, 0))[np.newaxis, :, :, :, ]
        return imgs

    def classic_test(self, path_list):
        assert len(path_list) == 14
        test_batch = self.read_files(path_list)
        cur_input = test_batch[:, :, :, :4]
        cur_input[cur_input > 80] = 0
        cur_input[cur_input < 15] = 0
        cur_input = normalization(cur_input)
        img_outs = []
        for img_index in range(10):
            test_output = self.sess.run(
                self.test_scale_preds[-1],
                feed_dict={
                    self.test_input_frame: cur_input,
                }
            )
            img_outs.append(test_output[0, :, :, 0])
            cur_input = np.concatenate([cur_input[:, :, :, -3:], test_output], axis=3)
        img_outs = np.array(img_outs)
        img_outs = denormalization(img_outs)
        img_outs[img_outs > 80] = 0
        img_outs[img_outs < 15] = 0
        return img_outs
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
                    test_output = self.sess.run(
                        self.test_scale_preds[-1],
                        feed_dict={
                            self.test_input_frame: test_input_frame,
                        }
                    )
                    img_out.append(test_output[0,:,:,0])
                    test_input_frame = np.concatenate([test_input_frame[:, :, :, -3:], test_output], axis=3)

                img_out = np.stack(img_out, 0)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1
        MSE = MSE / count
        return MSE

if __name__ == '__main__':
    import sys

    model_name = 'Multi_Scale_CNN'

    base_path = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(base_path, 'config', model_name + '.yml')
    configuration = yaml.load(open(config_path))

    # initialize the data iterator
    train_data_itertor = \
        SequenceRadarDataIterator(
            root_path=configuration['TRAIN_DATA_SAVE_DIR'],
            mode='Train',
            height=700,
            width=900,
        )

    test_data_itertor = \
        SequenceRadarDataIterator(
            root_path=configuration['TEST_DATA_SAVE_DIR'],
            mode='Test',
            height=700,
            width=900
        )

    valid_data_iterator = \
        SequenceRadarDataIterator(
            root_path=configuration['TRAIN_DATA_SAVE_DIR'],
            mode='Valid',
            height=700,
            width=900
        )

    # Initialize the model
    model = MutilScaleCNN(
        name = configuration['NAME'],
        train_iter = train_data_itertor,
        test_iter = test_data_itertor,
        valid_iter= valid_data_iterator,
        info = configuration,
    )
    # model.train()
    model.load_model()
    model.temp_test()
    # model.classic_test()
    # model.test()