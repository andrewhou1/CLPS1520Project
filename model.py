import tensorflow as tf


# TODO instead of adding only one channel for labels, the labels should be 1-hot vectors, so we need 3+num_channels
# layers to the input


class CNNModel:
    def __init__(self, hidden_size_1, hidden_size_2, patch_size, batch_size, num_classes, learning_rate):
        # TODO fix padding
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        print "params:", batch_size, patch_size, hidden_size_1, hidden_size_2, self.num_classes
        self.inpt = tf.placeholder(dtype=tf.float32, shape=[batch_size, patch_size, patch_size, 3+num_classes])
        print "**** input", self.inpt.get_shape()
        self.output = tf.placeholder(tf.int32, [1, 1])

        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, self.hidden_size_1], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size_1]))

        h_conv1 = tf.nn.conv2d(self.inpt, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        tanh = tf.tanh(h_pool1)
        print "**** tanh", tanh.get_shape()

        W_conv2 = tf.Variable(tf.truncated_normal([8, 8, self.hidden_size_1, self.hidden_size_2], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size_2]))

        h_conv2 = tf.nn.conv2d(tanh, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        print "&&&& h_conv2", h_conv2.get_shape()
        W_conv3 = tf.Variable(tf.truncated_normal([1, 1, self.hidden_size_2, self.num_classes], stddev=0.1))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

        h_conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
        print "&&&& h_conv3", h_conv3.get_shape()

        # figure out the frickin logits reshaping
        # h_conv3 shape is [batch_size x width x height x num_categories]
        conv3_shape = tf.shape(h_conv3)
        conv3_height = conv3_shape[1]
        conv3_width = conv3_shape[2]

        # TODO don't hardcode this slice
        center_pixel = tf.slice(h_conv3, begin=[0, conv3_height / 2, conv3_width / 2, 0],
                                size=[1, 1, 1, self.num_classes])
        self.logits = tf.reshape(center_pixel, [1, 1, self.num_classes])
        self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.output))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.error)


def save_model(sess, path, saver=None):
    """
    Saves a tensorflow session to the given path.
    NOTE: This currently saves *all* variables in the session, unless one passes in a custom Saver object.
    :param sess: The tensorflow session to save from
    :param path: The path to store the saved data
    :param saver: A custom saver object to use. This can be used to only save certain variables. If None,
    creates a saver object that saves all variables.
    :return: The saver object used.
    """
    if saver is None:
        saver = tf.train.Saver(tf.all_variables())
    saver.save(sess, path)
    return saver


def restore_model(sess, path, saver=None):
    """
    Loads a tensorflow session from the given path.
    NOTE: This currently loads *all* variables in the saved file, unless one passes in a custom Saver object.
    :param sess: The tensorflow checkpoint to load from
    :param path: The path to the saved data
    :param saver: A custom saver object to use. This can be used to only load certain variables. If None,
    creates a saver object that loads all variables.
    :return: The saver object used.
    """
    if saver is None:
        saver = tf.train.Saver(tf.all_variables())
    saver.restore(sess, path)
    return saver
