import tensorflow as tf
from  model import SRCNN
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 10, "Number of epoch")
flags.DEFINE_integer("image_size", 33, "The size of image input")
flags.DEFINE_integer("label_size", 21, "The size of image output")
flags.DEFINE_integer("c_dim", 3, "The size of channel")
flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("scale", 3, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 21, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4 , "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")



def main(_): #?
    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_size = FLAGS.image_size,
                      label_size = FLAGS.label_size,
                      c_dim = FLAGS.c_dim)
        srcnn.train(FLAGS)



if __name__=='__main__':
    tf.app.run() # parse the command argument , the call the main function
