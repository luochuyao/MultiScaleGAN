import tensorflow as tf
import numpy as np

from multi_scale_gan.tfutils import log10
import multi_scale_gan.msg_constants as c

def combined_loss(gen_frames, gt_frames, d_preds,is_w=False, lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2):
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

    loss = lam_lp * lp_loss(gen_frames, gt_frames, l_num)
    loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    if c.ADVERSARIAL: loss += lam_adv * adv_loss(d_preds, tf.ones([batch_size, 1]),is_w)

    return loss

def wbce_loss(preds,targets):

    return tf.squeeze(-1 * (tf.matmul(targets, preds, transpose_a=True) +
                            tf.matmul(1 - targets, 1 - preds, transpose_a=True)))

def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))


def rainfall_to_pixel(rainfall_intensity, a=None, b=None):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    if a is None:
        a = c.ZR_a
    if b is None:
        b = c.ZR_b
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def get_loss_weight_symbol(data):
    balancing_weights = c.BALANCING_WEIGHTS
    weights = tf.ones_like(data) * balancing_weights[0]
    thresholds = [rainfall_to_pixel(ele) for ele in c.THRESHOLDS]
    for i , threshold in enumerate(thresholds):
        weights = weights + (balancing_weights[i+1] - balancing_weights[i]) \
                  * tf.to_float(data >= threshold)
    return weights

def lp_loss(gen_frames, gt_frames, l_num):
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
        weight = get_loss_weight_symbol(gt_frames[i])

        scale_losses.append(tf.reduce_sum(weight*tf.abs(gen_frames[i] - gt_frames[i])**l_num))

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
    for i in range(len(gen_frames)):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(1), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_sum((grad_diff_x ** alpha + grad_diff_y ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))


def adv_loss(preds, labels,is_w = False):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.

    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).

    @return: The adversarial loss.
    """
    # calculate the loss for each scale
    scale_losses = []
    for i in range(len(preds)):
        if is_w:
            loss = wbce_loss(preds[i],labels)
        else:
            loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses))
