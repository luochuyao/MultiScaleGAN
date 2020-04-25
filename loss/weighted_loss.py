import numpy as np
import tensorflow as tf



def weighted_l2(pred, gt,info):
    weight = get_loss_weight_symbol(gt,info)
    l2 = weight * tf.square(pred - gt)
    l2 = tf.reduce_sum(l2)
    return l2


def get_loss_weight_symbol(data,info, seq_len=1):
    # data = np.array(data)
    if info['EVALUATE']['USE_BALANCED_LOSS']:
        balancing_weights = info['EVALUATE']['BALANCING_WEIGHTS']

        # print(data.shape)
        weights = tf.ones_like(data) * balancing_weights[0]
        # print(weights.shape)
        thresholds = [rainfall_to_pixel(ele,info) for ele in  info['EVALUATE']['THRESHOLDS']]
        for i , threshold in enumerate(thresholds):
            # print(data >= threshold)
            # mid = np.ones_like(data) * (balancing_weights[i + 1] - balancing_weights[i])
            # mid * tf.to_int32(data >= threshold)
            # print('汪')
            weights = weights + (balancing_weights[i+1] - balancing_weights[i]) \
                      * tf.cast(tf.to_float(data >= threshold),dtype=tf.float32)


    if info['EVALUATE']['TEMPORAL_WEIGHT_TYPE'] == "same":
        return weights
    else:
        raise NotImplementedError


def rainfall_to_pixel(rainfall_intensity, info,a=None, b=None):
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
        a = info['EVALUATE']['ZR_a']
    if b is None:
        b = info['EVALUATE']['ZR_b']
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals