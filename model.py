import numpy as np 
import tensorflow as tf


class objectModel:
	def __init__(self, hidden_size, patch_size, batch_size, num_classes, learning_rate):
		self.hidden_size = hidden_size
		self.patch_size = patch_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		
		self.inpt = tf.placeholder(tf.float32, [batch_size, patch_size, patch_size, 3]) 
		self.output = tf.placeholder(tf.int32, [1, 1])

		W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 3, 25]), stddev=0.1)
		b_conv1 = tf.Variable(tf.constant(0.1, shape = [25]))
		
		h_conv1 = tf.nn.conv2d(self.inpt, W_conv1, strides = [1, 2, 2, 1], padding='SAME')+b_conv1
		h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

		tanh = tf.tanh(h_pool1)
		
		W_conv2 = tf.Variable(tf.truncated_normal([8, 8, 25, 50]), stddev = 0.1)
		b_conv2 = tf.Variable(tf.constant(0.1, shape = [50]))

		h_conv2 = tf.nn.conv2d(tanh, W_conv2, strides = [1, 2, 2, 1], padding='SAME')+b_conv2
		
		W_conv3 = tf.Variable(tf.truncated_normal([1, 1, 50, num_classes], stddev = 0.1))
		b_conv3 = tf.Variable(tf.constant(0.1, shape = [num_classes]))
		
		h_conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides = [1, 2, 2, 1], padding = 'SAME')+b_conv3
		
		#figure out the frickin logits reshaping
		logits = tf.reshape(h_conv3, [1, 1, num_classes])
		error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.output))
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)		
		

		
