
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import os
os.environ['KMP_WARNINGS'] = 'off'
import multi_scale_gan.wmsg_constants as wc

import math
from multi_scale_gan.g_model import GeneratorModel
from multi_scale_gan.d_model import DiscriminatorModel
from data.radar_sequence_iterator import *
from multi_scale_gan.utils import *


class AVGRunner:
    def __init__(self, num_steps, model_load_path, num_test_rec):
        """
        Initializes the Adversarial Video Generation Runner.

        @param num_steps: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        @param num_test_rec: The number of recursive generations to produce when testing. Recursive
                             generations use previous generations as input to predict further into
                             the future.
        """

        self.global_step = 50000
        self.num_steps = num_steps
        self.num_test_rec = num_test_rec
        self.best_valid_res = math.inf
        self.test_limit = 1
        self.test_count = 0
        config = tf.ConfigProto(allow_soft_placement=True)

        self.sess = tf.Session(config=config)
        self.summary_writer = tf.summary.FileWriter(wc.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        if wc.ADVERSARIAL:
            print('Init discriminator...')
            self.d_model = DiscriminatorModel(self.sess,
                                              self.summary_writer,
                                              wc.TRAIN_HEIGHT,
                                              wc.TRAIN_WIDTH,
                                              wc.SCALE_CONV_FMS_D,
                                              wc.SCALE_KERNEL_SIZES_D,
                                              wc.SCALE_FC_LAYER_SIZES_D,
                                                  True)

        print('Init generator...')
        self.g_model = GeneratorModel(self.sess,
                                      self.summary_writer,
                                      wc.TRAIN_HEIGHT,
                                      wc.TRAIN_WIDTH,
                                      wc.FULL_HEIGHT,
                                      wc.FULL_WIDTH,
                                      wc.SCALE_FMS_G,
                                      wc.SCALE_KERNEL_SIZES_G,
                                      True)

        print('Init variables...')
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

        # initialize the data iterator
        self.train_data_itertor = \
            SequenceRadarDataIterator(
                root_path='/mnt/A/meteorological/RadarData/',
                mode='Train',
                height=700,
                width=900,
            )

        self.test_data_itertor = \
            SequenceRadarDataIterator(
                root_path='/mnt/A/meteorological/AllRadarData/',
                mode='Test',
                height=700,
                width=900
            )

        self.valid_data_iterator = \
            SequenceRadarDataIterator(
                root_path='/mnt/A/meteorological/RadarData/',
                mode='Valid',
                height=700,
                width=900
            )

        # if load path specified, load a saved model
        if model_load_path is not None:
            self.saver.restore(self.sess, model_load_path)
            print('Model restored from ' + model_load_path)
        else:
            pass
        self.sess.run(tf.global_variables_initializer())
    @property
    def vars(self):
        return  [var for var in tf.global_variables()]

    def count2(self):
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    def load_model(self,load_path):
        self.saver.restore(
            self.sess,
            load_path
        )
        print('model has been loaded')

    def save_model(self,save_path):

        self.saver.save(
            self.sess,
            save_path
        )
        print('model has been saved')

    def train(self):
        """
        Runs a training loop on the model networks.
        """
        for i in range(self.num_steps):
            if wc.ADVERSARIAL:
                # update discriminator
                batch = self.train_data_itertor.sample(wc.BATCH_SIZE)
                batch[batch>80.0] = 0
                batch[batch<15.0] = 0
                batch = normalize_frames(batch)
                # print('Training discriminator...')
                self.d_model.train_step(batch, self.g_model)

            # update generator
            batch = self.train_data_itertor.sample(wc.BATCH_SIZE)
            batch[batch > 80.0] = 0
            batch[batch < 15.0] = 0
            batch = normalize_frames(batch)
            # print('Training generator...')
            self.global_step = self.g_model.train_step(
                batch, discriminator=(self.d_model if wc.ADVERSARIAL else None))



            # save the models
            if self.global_step % wc.MODEL_SAVE_FREQ == 0:

                print('Saving models...')
                self.saver.save(self.sess,
                                wc.MODEL_SAVE_DIR + 'model.ckpt',
                                global_step=self.global_step)
                print('Saved models! save path is:',str(wc.MODEL_SAVE_DIR + 'model.ckpt'))

            # test generator model
            if self.global_step % wc.TEST_FREQ == 0:
                valid_res = self.valid()
                if valid_res < self.best_valid_res:
                    self.saver.save(self.sess,
                                    wc.MODEL_SAVE_DIR + 'model_best.ckpt')
                    print('Model improve')
                    self.best_valid_res = valid_res
                    self.test_count = 0
                else:
                    print('Model do not imporve')
                    self.test_count = self.test_count + 1

                if self.test_count > self.test_limit:
                    print('Model can not be imporved')
                    self.test()
                    break

    def valid(self):
        count = 0
        MSE = 0
        for i in range(self.valid_data_iterator.number_sample):
            frame_dat, names, file_name, = self.valid_data_iterator.sample_process(i)
            frame_dat[frame_dat > 80] = 0
            frame_dat[frame_dat < 15] = 0
            frame_dat = normalize_frames(frame_dat)
            for j in range((len(frame_dat) - 13) // 10):
                # print('current process ' + str(i) + '/' + str(
                #     self.valid_data_iterator.number_sample) + '\tfinish ' + str(
                #     100 * j / (len(frame_dat) - 13)) + '%')
                current_process = frame_dat[j * 5:j * 5 + 14]
                test_batch = current_process
                test_batch = test_batch.transpose((1, 2, 0))[np.newaxis, :, :, :]
                img_out = self.g_model.gen_pred(test_batch,10)
                tars = current_process[-10:]
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1
        MSE = MSE / count
        return MSE

    def test(self):
        count = 0
        MSE = 0
        clean_fold('/mnt/A/meteorological/2500_ref_seq/multi_scale_wgan_180000/')
        for i in range(self.test_data_itertor.number_sample):
            frame_dat, names, file_name, = self.test_data_itertor.sample_process(i)
            frame_dat[frame_dat > 80] = 0
            frame_dat[frame_dat < 15] = 0
            frame_dat = normalize_frames(frame_dat)
            clean_fold('/mnt/A/meteorological/2500_ref_seq/multi_scale_wgan_180000/' + file_name + '/')

            for j in range(len(frame_dat) - 23):
                print('current process ' + str(i) + '/' + str(
                    self.test_data_itertor.number_sample) + '\tfinish ' + str(
                    100 * j / (len(frame_dat) - 23)) + '%')
                current_names = names[j:j + 24]
                current_process = frame_dat[j:j + 24]
                test_batch = current_process[:14]
                test_batch = test_batch.transpose((1, 2, 0))[np.newaxis, :, :, :]
                tars = current_process[-10:]
                img_out = self.g_model.gen_pred(test_batch, 10)
                mse = np.mean(np.square(tars - img_out))
                MSE = MSE + mse
                count = count + 1
                img_out = denormalize_frames(img_out)
                clean_fold(
                    '/mnt/A/meteorological/2500_ref_seq/multi_scale_wgan_180000/' + file_name + '/process_' + str(j) + '/')
                clean_fold('/mnt/A/meteorological/2500_ref_seq/multi_scale_wgan_180000/' + file_name + '/process_' + str(
                    j) + '/predict/')
                for img_index in range(len(current_process) - 4):
                    if img_index < 10:
                        pass
                    else:
                        break
                    imsave('/mnt/A/meteorological/2500_ref_seq/multi_scale_wgan_180000/' + file_name + '/process_' + str(
                        j) + '/predict/pred_' + current_names[img_index + 4] + '.png',
                           img_out[img_index])

        MSE = MSE / count
        print('count is:', str(count), 'total mse is:', str(MSE))

        return MSE

    def read_files(self, path_list):
        imgs = []
        for path in path_list:
            img = imread(path)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs[imgs > 80] = 0
        imgs[imgs < 15] = 0
        imgs = normalize_frames(imgs)
        imgs = imgs.transpose((1, 2, 0))[np.newaxis, :, :, :]
        return imgs

    def classic_test(self, path_list):
        assert len(path_list) == 14
        test_batch = self.read_files(path_list)
        img_outs = self.g_model.gen_pred(test_batch, 10)
        img_outs = denormalize_frames(img_outs)
        return img_outs





def main():

    test_only = True
    num_test_rec = 1  # number of recursive predictions to make on test
    num_steps = 480000
    load_path = None
    runner = AVGRunner(num_steps, load_path, num_test_rec)
    load_path = '../model_lib/multi_scale_wgan/model.ckpt'
    runner.load_model(load_path)


    # runner.test()
    # if test_only:
    #     runner.test()
    # else:
    #     runner.train()



if __name__ == '__main__':

    main()
