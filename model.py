import tensorflow as tf
from utils import (
    input_setup,
    checkpoint_dir
)
class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size,
                 label_size,
                 c_dim):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.build_model()






    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        
        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, self.c_dim, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, self.c_dim], stddev=1e-3), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim], name='b3'))
        }
        
        self.pred = self.model()
        
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint



    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3'] # This layer don't need ReLU
        return conv3

    def train(self, config):
        
        # NOTE : if train, the nx, ny is ingnore, will be 0
        nx, ny = input_setup(config)
        
        
