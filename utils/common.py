import tensorflow as tf
import numpy as np
import os

def read_image(filepath):
    image = tf.io.decode_image(tf.io.read_file(filepath), channels=3)
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    return image

def write_image(filepath, src):
    tf.io.write_file(filepath, tf.io.encode_png(src))

# https://www.researchgate.net/publication/284923134
def rgb2ycbcr(src):
    if type(src) != np.ndarray:
        src = src.numpy()
    rgb = np.float32(src)
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    
    ycbcr = np.zeros(shape=rgb.shape)
    # *Intel IPP
    # ycbcr[:,:,0] = 0.257 * R + 0.504 * G + 0.098 * B + 16
    # ycbcr[:,:,1] = -0.148 * R - 0.291 * G + 0.439 * B + 128
    # ycbcr[:,:,2] = 0.439 * R - 0.368 * G - 0.071 * B + 128
    # *Intel IPP specific for the JPEG codec
    ycbcr[:,:,0] =  0.299 * R + 0.587 * G + 0.114 * B
    ycbcr[:,:,1] =  -0.16874 * R - 0.33126 * G + 0.5 * B + 128
    ycbcr[:,:,2] =  0.5 * R - 0.41869 * G - 0.08131 * B + 128
    
    # @Y in range [16, 235]
    ycbcr[:,:,0] = np.clip(ycbcr[:,:,0], 16, 235)
    # @Cb, Cr in range [16, 240]
    ycbcr[:,:,[1, 2]] = np.clip(ycbcr[:,:,[1, 2]], 16, 240)
    ycbcr = tf.cast(ycbcr, tf.uint8)
    return ycbcr

# https://www.researchgate.net/publication/284923134
def ycbcr2rgb(src):
    if type(src) != np.ndarray:
        src = src.numpy()
    ycbcr = np.float32(src)
    Y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    rgb = np.zeros(shape=ycbcr.shape)
    # *Intel IPP
    # rgb[:,:,0] = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    # rgb[:,:,1] = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    # rgb[:,:,2] = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    # *Intel IPP specific for the JPEG codec
    rgb[:,:,0] = Y + 1.402 * Cr - 179.456
    rgb[:,:,1] = Y - 0.34414 * Cb - 0.71414 * Cr + 135.45984
    rgb[:,:,2] = Y + 1.772 * Cb - 226.816

    rgb = np.clip(rgb, 0, 255)
    rgb = tf.cast(rgb, tf.uint8)
    return rgb

# list all file in dir and sort
def sorted_list(dir):
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls

def resize_bicubic(src, h, w):
    image = tf.image.resize(src, [h, w], method=tf.image.ResizeMethod.BICUBIC)
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image

# source: https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
def gaussian_blur(src, kernel_size=3, sigma=0.5):
    if len(src.shape) == 4 and src.shape[0] > 1:
        ValueError("src should be a single image, not a batch")

    def gaussian_kernel(channels, ksize, sigma):
        ax = tf.range(-ksize // 2 + 1.0, ksize // 2 + 1.0)
        x, y = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(x*x + y*y) / (2.0 * sigma*sigma))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    kernel = gaussian_kernel(src.shape[-1], kernel_size, sigma)
    kernel = kernel[..., tf.newaxis]
    image = tf.cast(src, tf.float32)
    image = tf.expand_dims(image, axis=0)
    blur_image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], padding='SAME')[0]
    blur_image = tf.cast(blur_image, tf.uint8)
    return blur_image
    
def upscale(src, scale):
    h = int(src.shape[0] * scale)
    w = int(src.shape[1] * scale)
    image = resize_bicubic(src, h, w)
    return image

def downscale(src, scale):
    h = int(src.shape[0] / scale)
    w = int(src.shape[1] / scale)
    image = resize_bicubic(src, h, w)
    return image

def norm01(src):
    return src / 255

def denorm01(src):
    return src * 255

def exists(path):
    return os.path.exists(path)

def PSNR(y_true, y_pred, max_val=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    MSE = tf.reduce_mean(tf.square(y_true - y_pred))
    return 10 * tf.math.log(max_val * max_val / MSE) / tf.math.log(10.0)

def random_crop(image, h, w, c=3):
    crop = tf.image.random_crop(image, [h, w, c])
    return crop

def random_transform(src):
    _90_left, _90_right, _180 = 1, 3, 2
    operations = {
        0 : (lambda x : x                                 ),
        1 : (lambda x : tf.image.rot90(x, k=_90_left)     ),
        2 : (lambda x : tf.image.rot90(x, k=_90_right)    ),
        3 : (lambda x : tf.image.rot90(x, k=_180)         ),
        4 : (lambda x : tf.image.random_flip_left_right(x)),
        5 : (lambda x : tf.image.random_flip_up_down(x)   ),
    }
    idx = np.random.choice([0, 1, 2, 3, 4, 5])
    image_transform = operations[idx](src)
    return image_transform

def shuffle(X, Y):
    if X.shape[0] != Y.shape[0]:
        ValueError("X and Y must have the same number of elements")
    indices = np.arange(0, X.shape[0])
    np.random.shuffle(indices)
    X = tf.gather(X, indices)
    Y = tf.gather(Y, indices)
    return X, Y
