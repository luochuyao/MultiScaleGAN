import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
from scipy.misc import imsave
import math
def log10(t):
    """
    Calculates the base-10 log of each element in t.

    @param t: The tensor from which to calculate the base-10 log.

    @return: A tensor with the base-10 log of each element in t.
    """

    numerator = tf.log(t)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

class BaseModel(object):

    def __init__(self,name,g_net,d_net,train_iter,test_iter,info):
        self.name = name
        self.g_net = g_net
        self.d_net = d_net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.width = 64
        self.height = 64
        self.best_mse = math.inf
        self.info = info
        self.epoches = self.info['TRAINING']['EPOCHES']
        self.train_batch_size = self.info['TRAINING']['BATCH_SIZE']
        self.test_batch_size = self.info['TESTING']['BATCH_SIZE']


    def save_sequence_img(self,sequence_imgs,save_fold):
        for i,img in enumerate(sequence_imgs):
            save_path = save_fold+'t_'+str(i)+'.png'
            imsave(save_path,img)

    def save_gif(self,sequence_imgs,save_path):
        frames = []
        for img in sequence_imgs:
            frames.append(img)
        imageio.mimsave(save_path,frames,'GIF',duration = 0.5)

    def show_train_hist(self,hist, show=False, save=False, path='Train_hist.png'):
        x = range(len(hist['D_losses']))

        y1 = hist['D_losses']
        y2 = hist['G_losses']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        if save:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()