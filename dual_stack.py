import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from prepare import readbunchobj
from sklearn.preprocessing import MinMaxScaler


class DualStackAutoEncoder(object):

    def __init__(self, learning_rate, training_epochs, batch_sizes):


