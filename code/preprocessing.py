import tensorflow as tf
from GAN import GAN_MODEL
from GAN import d_loss, g_loss, d_acc_fake, d_acc_real, g_acc, get_dis_model, get_gen_model

import os
import tensorflow as tf
import numpy as np
import random
import math
from matplotlib import pyplot as plt


# ensures that we run only on cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

'''if "X0" not in globals():
    (X0, L0), (_, _) = tf.keras.datasets.mnist.load_data()
    X0 = tf.cast(X0, tf.float32) / 255.0
    X0 = tf.expand_dims(X0, -1)'''

# We'll use all the available examples from both the training and test
# sets.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_labels = np.concatenate([y_train, y_test])



def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset 
    Returns:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when 
    the ground truth label for image i is j, and targets[i, :j] & 
    targets[i, j + 1:] are equal to 0
    """
    targets = np.zeros((labels.shape[0], class_size))
    for i, label in enumerate(labels):
        targets[i, label] = 1
    targets = tf.convert_to_tensor(targets)
    targets = tf.cast(targets, tf.float32)
    return targets

# Scale the pixel values to [0, 1] range, add a channel dimension to
# the images, and one-hot encode the labels.
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
all_labels = one_hot(all_labels, 10)

# Create tf.data.Dataset.
dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {all_digits.shape}")
print(f"Shape of training labels: {all_labels.shape}")