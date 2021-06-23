import keras.layers
import numpy as np
import tensorflow as tf
import os


def check_shape(shape):
    if isinstance(shape[0], float):
        shape[0] = int(shape[0])
    if isinstance(shape[1], float):
        shape[1] = int(shape[1])


def get_wave_kernel(shape):
    check_shape(shape)
    mat_hp = np.zeros((shape[0], shape[1]))
    mat_lp = np.zeros((shape[0], shape[1]))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    lp_filter = np.load(BASE_DIR + '/../utils/wave/db4/lp.npy')
    hp_filter = np.load(BASE_DIR + '/../utils/wave/db4/hp.npy')

    for i in range(shape[1]):
        for j in range(8):
            mat_lp[2 * i - j, i] = lp_filter[j]
            mat_hp[2 * i - j, i] = hp_filter[j]

    return mat_lp, mat_hp


def _wave_variable_on_cpu(matr, name, trainable):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.Variable(matr, trainable=trainable, name=name, dtype=tf.float32)
        # shape = matr.shape
        # var = tf.get_variable(name, shape)
    return var


def variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        with tf.variable_scope("variable_on_cpu", reuse=tf.AUTO_REUSE):
            dtype = tf.float16 if use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def wave_variable_with_l1(matr, name, wd, l1_value, sim_reg=None):
    """Helper to create an wavelt initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      l1_value: add L1Loss to wavelet initialized weight
      sim_reg: regularization terms on forcing trained weight to be similar with the initial weight

    Returns:
      Variable Tensor
    """
    var = _wave_variable_on_cpu(
        matr,
        name=name, trainable=True)
    var_reg = _wave_variable_on_cpu(
        matr,
        name=name, trainable=False)

    if l1_value is not None:
        l1_loss = tf.multiply(tf.reduce_mean(tf.abs(var)), l1_value, name='l1_loss')
        tf.add_to_collection('losses', l1_loss)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    if sim_reg is not None:
        similar_reg = tf.multiply(tf.nn.l2_loss(var - var_reg), sim_reg, name='similar_reg_loss')
        tf.add_to_collection('losses', similar_reg)
    return var


def tf_concat(axis, values):
    return tf.concat(values, axis)


class AnchorsLayer(keras.layers.Layer):
    def __init__(self, anchors):
        super(AnchorsLayer, self).__init__()
        self.anchors_v = tf.Variable(anchors)

    def call(self):
        return self.anchors_v