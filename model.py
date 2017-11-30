import tensorflow as tf
import numpy as np
import time
import os

from utils import (
    input_setup,
    checkpoint_dir,
    read_data,
    merge,
    checkimage,
    imsave
)
class ESPCN(object):

    def __init__(self,
                 sess,
                 image_size,
#                 label_size,
                 scale,
                 c_dim):
        self.sess = sess
        self.image_size = image_size
#        self.label_size = label_size
        self.c_dim = c_dim
        self.scale = scale
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.images_test = tf.placeholder(tf.float32,[None])
        self.labels = tf.placeholder(tf.float32, [None, self.image_size * self.scale , self.image_size * self.scale, self.c_dim], name='labels')
        
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, self.c_dim, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=1e-3), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b3'))
        }
        
        self.pred = self.model()
        
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint

    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        print(conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        print(conv2)
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3'] # This layer don't need ReLU
        print(conv3)

        ps = self.PS(conv3, self.scale)
        return tf.nn.tanh(ps)

    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        print(X)
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        print(X)
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        print(X)
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        print(X)
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        print(X)
        return tf.reshape(X, (1, a*r, b*r, 1))

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        return X

    def train(self, config):
        
        # NOTE : if train, the nx, ny are ingnored
        nx, ny = input_setup(config)

        data_dir = checkpoint_dir(config)
        
        input_, label_ = read_data(data_dir)
        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        counter = 0
        time_ = time.time()

        self.load(config.checkpoint_dir)
        # Train
        if config.is_train:
            print("Now Start Training...")
            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = len(input_) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = input_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    batch_labels = label_[idx * config.batch_size : (idx + 1) * config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % ((ep+1), counter, time.time()-time_, err))
                        #print(label_[1] - self.pred.eval({self.images: input_})[1],'loss:]',err)
                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)
        # Test
        else:
            print("Now Start Testing...")
            input_test = input_[15].reshape(1,17,17,3)
            result = self.pred.eval({self.images_test: input_test})
            x = np.squeeze(result)
            print(x.shape)
            checkimage(input_[15])
            checkimage(x)
            checkimage(label_[15])
            #image_LR = merge(input_, [nx, ny], self.c_dim)
            #imsave(image, config.result_dir+'/result.png', config)

    def load(self, checkpoint_dir):
        """
            To load the checkpoint use to test or pretrain
        """
        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s_%s" % ("espcn", self.image_size)# give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        # Check the checkpoint is exist 
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")
    def save(self, checkpoint_dir, step):
        """
            To save the checkpoint use to test or pretrain
        """
        model_name = "ESPCN.model"
        model_dir = "%s_%s" % ("espcn", self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
