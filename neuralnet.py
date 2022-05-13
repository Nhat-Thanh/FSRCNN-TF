from tensorflow.keras.layers import Conv2D, Input, PReLU, Conv2DTranspose
from tensorflow.keras.initializers import RandomNormal, HeNormal
from tensorflow.keras.models import Model
import tensorflow as tf


def FSRCNN(scale):
    d = 56 
    s = 12
    m = 4

    X_in = Input(shape=(None, None, 3))
    X = Conv2D(filters=d, kernel_size=5, padding='same',
               kernel_initializer=HeNormal())(X_in)
    X = PReLU(shared_axes=[1, 2])(X)
    X = Conv2D(filters=s, kernel_size=1, padding='same',
               kernel_initializer=HeNormal())(X)
    X = PReLU(shared_axes=[1, 2])(X)
    
    for _ in range(0, m):
        X = Conv2D(filters=s, kernel_size=3, padding='same',
                   kernel_initializer=HeNormal())(X)
    X = PReLU(shared_axes=[1, 2])(X)

    X = Conv2D(filters=d, kernel_size=1, padding='same',
               kernel_initializer=HeNormal())(X)
    X = PReLU(shared_axes=[1, 2])(X)

    X = Conv2DTranspose(filters=3, kernel_size=9, strides=scale, padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(X)

    X_out = tf.clip_by_value(X, 0.0, 1.0)

    return Model(X_in, X_out)

