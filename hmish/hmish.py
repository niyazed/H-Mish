"""Tensorflow-Keras Implementation of HMish"""

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class HMish(Activation):
    '''
    Hard Mish - Memory Efficient and faster equivalent of Mish.
    .. math::
        hmish(x) = (x/2) * minimum(2, maximum(0, x+2))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('HMish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(HMish, self).__init__(activation, **kwargs)
        self.__name__ = 'HMish'


def hmish(inputs):
    return (inputs/2.0)* tf.minimum(2.0, tf.maximum(0.0, inputs+2.0))

get_custom_objects().update({'HMish': HMish(hmish)})