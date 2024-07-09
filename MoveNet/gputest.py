import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from tqdm import tqdm


# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)