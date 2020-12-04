# 两层的堆叠自编码


# 堆叠自编码器

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class DualStackedAutoEncoder(object):

    def __init__(self, list1, eta=0.01):
        """

        :param list1: [input_dimension, hidden_layer_1, ... , hidden_layer_n]
        :param eta:
        """
        