import tensorflow as tf
import numpy as np
from loss.weighted_loss import weighted_l2

def mnist_combined_loss(gen_frames, gt_frames):
    loss_lp = weight_scale_mse(gen_frames,gt_frames)
    loss_gdl = gdl_loss(gen_frames,gt_frames,1)
    loss = loss_lp+loss_gdl
    return loss

def weight_generator(data):

        balancing_weights = [1, 1, 2, 5, 10, 30]

        # print(data.shape)
        weights = tf.ones_like(data) * balancing_weights[0]
        # print(weights.shape)

        for i , threshold in enumerate([0.1,0.3,0.5,0.7,0.9]):
            weights = weights + (balancing_weights[i+1] - balancing_weights[i]) \
                      * tf.cast(tf.to_float(data >= threshold),dtype=tf.float32)
        return weights

def weight_scale_mse(gen_frames, gt_frames):
    loss = []
    for i in range(len(gen_frames)):
        current_loss = tf.losses.mean_squared_error(gen_frames[i],gt_frames[i])
        loss.append(current_loss)
    loss = tf.reduce_mean(tf.stack(loss))
    return loss


def combined_loss(gen_frames, gt_frames, info,lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2):
    """
    Calculates the sum of the combined adversarial, lp and GDL losses in the given proportion. Used
    for training the generative model.

    @param gen_frames: A list of tensors of the generated frames at each scale.
    @param gt_frames: A list of tensors of the ground truth frames at each scale.
    @param d_preds: A list of tensors of the classifications made by the discriminator model at each
                    scale.
    @param lam_adv: The percentage of the adversarial loss to use in the combined loss.
    @param lam_lp: The percentage of the lp loss to use in the combined loss.
    @param lam_gdl: The percentage of the GDL loss to use in the combined loss.
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @param alpha: The power to which each gradient term is raised in GDL loss.

    @return: The combined adversarial, lp and GDL losses.
    """
    batch_size = tf.shape(gen_frames[0])[0]  # variable batch size as a tensor

    loss_lp=lam_lp * lp_loss(gen_frames, gt_frames, l_num,info)
    loss_gdl=lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)

    loss = loss_lp+loss_gdl

    return loss


def lp_loss(gen_frames, gt_frames, l_num,info):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    # calculate the loss for each scale
    scale_losses = []
    for i in range(len(gen_frames)):
        # scale_losses.append(tf.reduce_sum(tf.abs(gen_frames[i] - gt_frames[i]) ** l_num))
        # scale_losses.append(tf.reduce_sum(tf.multiply(tf.abs(gen_frames[i] - gt_frames[i])**l_num,((gt_frames[i]+1.1)*10))))
        # scale_losses.append(tf.reduce_sum(tf.multiply(tf.abs(gen_frames[i] - gt_frames[i])**l_num,((gt_frames[i]+1)*10))))
        # scale_losses.append(tf.reduce_sum(tf.multiply(tf.abs(gen_frames[i] - gt_frames[i])**l_num,(tf.maximum(-0.48,gt_frames[i]*69+33.29))**4/27000+1)))
        # scale_losses.append(tf.reduce_sum(tf.multiply(tf.abs(gen_frames[i] - gt_frames[i]) ** l_num,
        #                                               (tf.maximum(-0.48, gt_frames[i] * 69 + 33.29)) ** 4 / 27000 + 1)))
        if info['EVALUATE']['USE_BALANCED_LOSS']:
            # scale_losses.append(tf.reduce_sum(tf.multiply(tf.abs(gen_frames[i] - gt_frames[i])**l_num,tf.minimum(tf.maximum(-0.48,gt_frames[i]*69+33.29)**6/2430000+1,30))))
            scale_losses.append(weighted_l2(gen_frames[i], gt_frames[i],info))
        else:
            scale_losses.append(tf.reduce_sum(tf.multiply(tf.abs(gen_frames[i] - gt_frames[i])**l_num,tf.minimum(tf.maximum(-0.48,gt_frames[i]*69+33.29)**6/2430000+1,30))))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))


def gdl_loss(gen_frames, gt_frames, alpha):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.

    @return: The GDL loss.
    """
    # calculate the loss for each scale
    scale_losses = []
    # for i in range(len(gen_frames)):
    #     print(gen_frames[i].shape,gt_frames[i].shape)
    for i in range(len(gen_frames)):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(1), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'
        # print(gen_frames[i].shape, gt_frames[i].shape,filter_x.shape,filter_y.shape)
        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))
