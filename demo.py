import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import FSRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',      type=int, default=2,                   help='-')
parser.add_argument("--ckpt-path",  type=str, default="",                  help='-')
parser.add_argument("--image-path", type=str, default="dataset/test1.png", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    ValueError("scale must be 2, 3, or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/x{scale}/FSRCNN-x{scale}.h5"

sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    write_image("bicubic.png", bicubic_image)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    lr_image = rgb2ycbcr(lr_image)
    lr_image = norm01(lr_image)
    lr_image = tf.expand_dims(lr_image, axis=0)

    model = FSRCNN(scale)
    model.load_weights(ckpt_path)
    sr_image = model.predict(lr_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)
    sr_image = ycbcr2rgb(sr_image)
    write_image("sr.png", sr_image)

if __name__ == "__main__":
    main()
