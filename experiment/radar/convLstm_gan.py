import sys
import os

# runing the progress by nohup
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

import math
from loss.loss_functions import *
from util.ops import *
from data.radar_sequence_iterator import SequenceRadarDataIterator
import yaml
from model.FullConvRNN import *
from model.FullConv import *
from util.utils import *
from datetime import datetime

class SequenceBaseModel(object):
    def __init__(self,
                 epoches,
                 name,
                 gen_conv_rnn_nets,
                 gen_conv_nets,
                 dis_conv_nets,
                 dis_pool_nets,
                 train_data_iter,
                 test_data_iter,
                 valid_data_iter,
                 info,
                 train_batch_size,
                 test_batch_size
                 ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.name = name
        self.epoches = epoches
        self.train_data_iter = train_data_iter
        self.test_data_iter = test_data_iter
        self.valid_data_iter = valid_data_iter
        self.gen_conv_rnn_nets = gen_conv_rnn_nets
        self.gen_conv_nets = gen_conv_nets
        self.dis_conv_nets = dis_conv_nets
        self.dis_pool_nets = dis_pool_nets
        self.info = info
        self.layer_num = len(self.gen_conv_rnn_nets)
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

        self.train_input_frames = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.train_batch_size,
                self.info['TRAINING']['INPUT_SEQ_LEN'],
                self.info['TRAINING']['HEIGHT_TRAIN'],
                self.info['TRAINING']['WIDTH_TRAIN'],
                1
            ]
        )
        self.train_target_frames = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.train_batch_size,
                self.info['TRAINING']['HEIGHT_TRAIN'],
                self.info['TRAINING']['WIDTH_TRAIN'],
                1
            ]
        )

        self.test_input_frames = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.test_batch_size,
                self.info['TESTING']['INPUT_SEQ_LEN'],
                self.info['TESTING']['HEIGHT_TEST'],
                self.info['TESTING']['WIDTH_TEST'],
                1
            ]
        )
        self.test_target_frames = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.test_batch_size,
                self.info['TESTING']['HEIGHT_TEST'],
                self.info['TESTING']['WIDTH_TEST'],
                1
            ]
        )



class SequenceModel(SequenceBaseModel):

    def __init__(self,
                 epoches,
                 name,
                 gen_conv_rnn_nets,
                 gen_conv_nets,
                 dis_conv_nets,
                 dis_pool_nets,
                 train_data_iter,
                 test_data_iter,
                 valid_data_iter,
                 info,
                 train_batch_size,
                 test_batch_size
                 ):
        super(SequenceModel, self).__init__(
            epoches,
            name,
            gen_conv_rnn_nets,
            gen_conv_nets,
            dis_conv_nets,
            dis_pool_nets,
            train_data_iter,
            test_data_iter,
            valid_data_iter,
            info,
            train_batch_size,
            test_batch_size
        )
        self.scale_fully_connect_layer_sizes = [512, 64, 16, 1]
        self.test_step = self.info['TRAINING']['TEST_STEP']
        self.train_output_frames = self.build_generator_model(self.train_input_frames)
        self.test_output_frames = self.build_generator_model(self.test_input_frames)
        self.loss = combined_loss([self.train_output_frames], [self.train_target_frames],self.info)

        self.d_fake, self.d_real = self.build_discriminator_model(self.train_output_frames, self.train_target_frames)

        self.d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real, labels=tf.ones_like(self.d_real))
        ) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.zeros_like(self.d_fake))
        )

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake, labels=tf.ones_like(self.d_fake))
        )

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate_init = self.info['TRAINING']['LEARNING_RATE']
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate=learning_rate_init,
        #     global_step=self.global_step,
        #     decay_steps=5000,
        #     decay_rate=0.9,
        #     staircase=True
        # )
        self.d_rmsprop = tf.train.GradientDescentOptimizer(
            learning_rate=self.info['MODEL_PARAMETER_D']['LEARNING_RATE_D']) \
            .minimize(self.d_loss, var_list=self.dis_vars)
        self.g_rmsprop = tf.train.GradientDescentOptimizer(
            learning_rate=self.info['MODEL_PARAMETER_G']['LEARNING_RATE_G']) \
            .minimize(self.g_loss, var_list=self.gen_vars)
        self.adam_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.info['TRAINING']['LEARNING_RATE']
        ).minimize(
            self.loss,
            var_list=self.gen_vars
        )
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        print('initalize finnish')

    @property
    def gen_vars(self):
        return [var for var in tf.global_variables() if 'generator' in var.name]

    @property
    def dis_vars(self):
        return [var for var in tf.global_variables() if 'discriminator' in var.name]

    def train(self):
        tolerate_iter = 0
        best_loss = math.inf

        start_time = datetime.now()
        print('start training!!!')

        for step in range(self.epoches):
            # data loading
            frame_dat = self.train_data_iter.sample(
                batch_size=self.info['TRAINING']['BATCH_SIZE']
            )
            # data preprocess
            frame_dat[frame_dat < 15] = 0
            frame_dat[frame_dat > 80] = 0
            frame_dat = normalization(frame_dat)
            input_frames = frame_dat[:, :, :, :4]
            target_frames = frame_dat[:, :, :, 4:]
            input_frames = np.expand_dims(input_frames.transpose((0, 3, 1, 2)), axis=-1)
            target_frames = np.expand_dims(target_frames.transpose((0, 3, 1, 2)), axis=-1)[:, 0]

            #train data
            for _ in range(4):
                self.sess.run(
                    self.adam_optimizer,
                    feed_dict={
                        self.train_input_frames: input_frames,
                        self.train_target_frames: target_frames
                    }
                )

            # train discriminator
            self.sess.run(self.d_rmsprop, feed_dict={
                self.train_input_frames: input_frames,
                self.train_target_frames: target_frames
            })

            # train generator
            self.sess.run(self.g_rmsprop, feed_dict={
                self.train_input_frames: input_frames,
                self.train_target_frames: target_frames
            })

            # print the training information
            if (step + 1) % self.info['TRAINING']['DISTPLAY_STEP'] == 0:
                loss = self.sess.run(
                    self.loss,
                    feed_dict={
                        self.train_input_frames: input_frames,
                        self.train_target_frames: target_frames
                    }
                )
                end_time = datetime.now()
                print('*' * 50)
                print('step is :', str(step + 1))
                print('train batch loss is:', str(loss))
                print('time consume is:', (end_time - start_time).seconds)
                print('*' * 50)
                start_time = datetime.now()

            # validation
            if (step + 1) % self.test_step == 0:
                valid_loss = self.valid()
                if valid_loss>best_loss:
                    best_loss = valid_loss
                    tolerate_iter = 0
                    self.save_model()
                else:
                    tolerate_iter += 1
                    if tolerate_iter==self.info['TRAINING']['LOSS_LIMIT']:
                        print('the best MSE is:', best_loss)
                        self.load_model()
                        self.test()
                        break

    #saving model
    def save_model(self,model_name = None):

        model_fold = os.path.join(self.base_path,self.info['MODEL_SAVE_DIR'])

        if not os.path.exists(model_fold):
            os.mkdir(model_fold)
        else:
            pass
        if model_name == None:
            self.saver.save(
                self.sess,
                os.path.join(model_fold,'radar_model.ckpt')
            )

        else:
            self.saver.save(
                self.sess,
                os.path.join(model_fold, model_name)
            )
        print('model saved')

    # loading model
    def load_model(self,model_name = None):
        model_fold = os.path.join(self.base_path,self.info['MODEL_SAVE_DIR'])
        if not os.path.exists(model_fold):
            raise ('the path of model is null')
        else:
            if model_name == None:
                self.saver.restore(
                    self.sess,
                    os.path.join(model_fold,'radar_model.ckpt')
                )
            else:
                self.saver.restore(
                    self.sess,
                    os.path.join(model_fold, model_name)
                )
        print('model has been loaded')


    # build TF graph of generator
    def build_generator_model(self, input_frames):
        with tf.variable_scope('generator'):
            for idx, net in enumerate(self.gen_conv_rnn_nets):

                if idx == self.layer_num - 1:
                    conv_rnn_output = net(input_frames)
                else:
                    current_output = net(input_frames)

                    input_frames = current_output

            conv_input = conv_rnn_output

            for idx, net in enumerate(self.gen_conv_nets):
                current_output = net(conv_input)
                conv_input = current_output

            output_frames = current_output

            return output_frames

    # build TF graph of discriminator
    def build_discriminator_model(self,input_fake_frames,input_true_frames):

        with tf.variable_scope('discriminator'):
            input_frames = tf.concat([input_fake_frames,input_true_frames],axis=0)
            batch_size = input_frames.get_shape().as_list()[0]
            for idx, conv_net in enumerate(self.dis_conv_nets):
                pool_net = self.dis_pool_nets[idx]

                if idx == len(self.dis_conv_nets) - 1:
                    conv_output = pool_net(conv_net(input_frames))

                else:
                    conv_output = conv_net(input_frames)
                    current_output = pool_net(conv_output)

                    input_frames = current_output

            shape = conv_output.get_shape().as_list()

            preds = tf.reshape(conv_output, [-1, shape[1] * shape[2] * shape[3]])
            self.scale_fully_connect_layer_sizes.insert(0,int(shape[1] * shape[2] * shape[3]))

            for i in range(len(self.scale_fully_connect_layer_sizes) - 1):
                if i == len(self.scale_fully_connect_layer_sizes) - 2:
                    preds = linear(
                        input=preds,
                        w_shape=[
                            int(self.scale_fully_connect_layer_sizes[i]),
                            int(self.scale_fully_connect_layer_sizes[i + 1])
                        ],
                        b_shape=[
                            self.scale_fully_connect_layer_sizes[i + 1]
                        ],
                        b_const=0.1,
                        dtype=tf.float32,
                        activate='sigmoid'
                    )
                else:
                    preds = linear(
                        input=preds,
                        w_shape=[
                            int(self.scale_fully_connect_layer_sizes[i]),
                            int(self.scale_fully_connect_layer_sizes[i + 1])
                        ],
                        b_shape=[
                            self.scale_fully_connect_layer_sizes[i + 1]
                        ],
                        b_const=0.1,
                        dtype=tf.float32,
                        activate='relu'
                    )

            fake = preds[:int(batch_size/2),:]
            true = preds[int(batch_size/2):,:]
            return fake,true

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
                test_input_frame = test_input_frame[np.newaxis, :, :, :, np.newaxis]

                tars = current_process[-10:]
                img_out = []
                for img_index in range(len(current_process) - 4):
                    test_output = self.sess.run(
                        self.test_output_frames,
                        feed_dict={
                            self.test_input_frames: test_input_frame,
                        }
                    )
                    img_out.append(test_output[0, :, :, 0])
                    test_output = test_output[:, np.newaxis, :, :, :, ]
                    test_input_frame = np.concatenate([test_input_frame[:, -3:, :, :, :], test_output], axis=1)
                img_out = np.stack(img_out, 0)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1
        MSE = MSE / count
        return MSE

    # evaluate the model in terms of MSE
    def test(self):

        count = 0
        MSE = 0
        for i in range(self.test_data_iter.number_sample):
            frame_dat, names, file_name, = self.test_data_iter.sample_process(i)
            frame_dat[frame_dat > 80] = 0
            frame_dat[frame_dat < 15] = 0
            frame_dat = normalization(frame_dat)
            for j in range(len(frame_dat) - 13):
                print('current process ' + str(i) + '/' + str(
                    self.test_data_iter.number_sample) + '\tfinish ' + str(
                    100 * j / (len(frame_dat) - 13)) + '%')
                current_process = frame_dat[j:j + 14]

                test_input_frame = current_process[:4]
                test_input_frame = test_input_frame[np.newaxis, :, :, :, np.newaxis]

                tars = current_process[-10:]
                img_out = []
                for img_index in range(len(current_process) - 4):
                    test_output = self.sess.run(
                        self.test_output_frames,
                        feed_dict={
                            self.test_input_frames: test_input_frame,
                        }
                    )
                    img_out.append(test_output[0, :, :, 0])
                    test_output = test_output[:, np.newaxis, :, :, :, ]
                    test_input_frame = np.concatenate([test_input_frame[:, -3:, :, :, :], test_output], axis=1)
                img_out = np.stack(img_out, 0)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1

        MSE = MSE / count

        return MSE


if __name__ == '__main__':

    import sys

    model_name = 'ConvLSTM_GAN'

    base_path = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(base_path, 'config', model_name + '.yml')
    configuration = yaml.load(open(config_path))
    gen_conv_rnn_nets = []
    gen_conv_nets = []
    dis_conv_nets = []
    dis_pool_nets = []


    for idx, cell in enumerate(configuration['GENERATOR_MODEL_NETS']['CELLS']):
        cell_param = get_cell_param(cell)
        if idx == len(configuration['GENERATOR_MODEL_NETS']['CELLS']) - 1:
            net = Conv2DLSTM(
                name='generator_conv_lstm_layer_' + str(idx),
                cell_param=cell_param,
            )
        else:
            net = Conv2DLSTM(
                name='generator_conv_lstm_layer_' + str(idx),
                cell_param=cell_param,
                return_sequence=True
            )

        gen_conv_rnn_nets.append(net)

    for idx, param in enumerate(configuration['GENERATOR_MODEL_NETS']['CONVS']):
        if idx == len(configuration['GENERATOR_MODEL_NETS']['CONVS']) - 1:
            conv_param = get_conv_param(param, activate='tanh')
            net = Conv2D('generator_conv_layer_' + str(idx), conv_param)
        else:
            conv_param = get_conv_param(param)
            net = Conv2D('generator_conv_layer_' + str(idx), conv_param)
        gen_conv_nets.append(net)

    for idx,param in enumerate(configuration['DISCRIMINATOR_MODEL_NETS']['CONVS']):
        if idx == len(configuration['DISCRIMINATOR_MODEL_NETS']['CONVS'])-1:
            conv_param = get_conv_param(param,activate='sigmoid')
            net = Conv2D('discriminator_conv_layer_' + str(idx),conv_param)
        else:
            conv_param = get_conv_param(param)
            net = Conv2D('discriminator_conv_layer_' + str(idx),conv_param)
        dis_conv_nets.append(net)

    for idx,param in enumerate(configuration['DISCRIMINATOR_MODEL_NETS']['POOLS']):
        if idx == len(configuration['DISCRIMINATOR_MODEL_NETS']['POOLS'])-1:
            pool_param = get_pool_param(param)
            net = Pooling(pool_param)
        else:
            pool_param = get_pool_param(param)
            net = Pooling(pool_param)
        dis_pool_nets.append(net)
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
    model = SequenceModel(
        name=configuration['NAME'],
        epoches=configuration['TRAINING']['EPOCHES'],
        train_data_iter=train_data_itertor,
        test_data_iter=test_data_itertor,
        valid_data_iter=valid_data_iterator,
        gen_conv_rnn_nets=gen_conv_rnn_nets,
        gen_conv_nets=gen_conv_nets,
        dis_conv_nets = dis_conv_nets,
        dis_pool_nets = dis_pool_nets,
        info=configuration,
        train_batch_size=configuration['TRAINING']['BATCH_SIZE'],
        test_batch_size=configuration['TESTING']['BATCH_SIZE']
    )

    model.train()