import tensorflow as tf
from  model import SRCNN
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 10, "Number of epoch")
flags.DEFINE_integer("image_size", 33, "The size of image input")
flags.DEFINE_integer("label_size", 21, "The size of image output")
flags.DEFINE_integer("c_dim", 3, "The size of channel")




def main(_): #?
    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_size = FLAGS.image_size,
                      label_size = FLAGS.label_size,
                      c_dim = FLAGS.c_dim)





if __name__=='__main__':
    tf.app.run() # parse the command argument , the call the main function
