

from math import *
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt

def FK(angles):
	l1 = 0.2
	l2 = 0.2
	x = l1*cos(angles[0]) + l2*cos(angles[0] + angles[1])
	y = l1*sin(angles[0]) + l2*sin(angles[0] + angles[1])
	return np.array([x, y])

def generateData(batchSize):
	batch_x = pi*np.random.rand(batchSize, 2)
	batch_y = np.zeros((batchSize, 2))
	batch_l = np.ones((batchSize,2)) * 0.2
	
	for i in range(batchSize):
		batch_y[i, :] = FK(batch_x[i,:])

	return batch_x, batch_y, batch_l	

if __name__ == '__main__':
	tf.reset_default_graph()

	# Parameters
	batchSize = 100
	
	# Network Parameters
	numInput = 2
	numOutput = 2

	# tf Graph input
	X = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
	L = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
	Y = tf.placeholder(shape=[None, numOutput], dtype=tf.float64)
	#L = tf.constant([[0.2, 0.2]], dtype=tf.float64)

	# Construct model
	h_jointAngles = layers.fully_connected(inputs=X, num_outputs=100, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.tanh) # Change sigmoid (relu, cos/sin)   |    is there bias ??
	h_lengths = layers.fully_connected(inputs=L, num_outputs=100, biases_initializer=tf.zeros_initializer(), activation_fn=None)
	new_input = tf.concat([h_jointAngles, h_lengths], 1)
	outputLayer = layers.fully_connected(inputs=new_input, num_outputs=numOutput, biases_initializer=tf.zeros_initializer(), activation_fn=None)

	init = tf.global_variables_initializer()
	# Later, launch the model, use the saver to restore variables from disk, and
	# do some work with the model.
	with tf.Session() as sess:

		sess.run(init)
		# Restore variables from disk.
		saver = tf.train.import_meta_graph('./model/checkpoint.ckpt.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./model/'))
		print("Model restored.")
		# Check the values of the variables
		batch_x, batch_y, batch_l = generateData(batchSize)
		out = sess.run(outputLayer, feed_dict={X: batch_x, L:batch_l})
		# print(out.shape)
		# print(batch_y.shape)
		error = (out - batch_y)**2
		print(np.sum(error))
		print(tf.reduce_mean((out - batch_y)**2))