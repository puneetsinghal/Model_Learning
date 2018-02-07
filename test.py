

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
	
	for i in range(batchSize):
		batch_y[i, :] = FK(batch_x[i,:])

	return batch_x, batch_y

if __name__ == '__main__':
	tf.reset_default_graph()

	# Parameters
	learningRate = 0.01
	numSteps = 100000
	batchSize = 1000
	displayStep = 100

	# Network Parameters
	numInput = 2
	numOutput = 2

	# tf Graph input
	X = tf.placeholder(shape=[1,numInput], dtype=tf.float64)

	# Construct model
	h0 = layers.fully_connected(inputs=X, num_outputs=100, activation_fn=tf.nn.sigmoid)
	h1 = layers.fully_connected(inputs=h0, num_outputs=numOutput, activation_fn=None)
	
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
		batch_x, batch_y = generateData(1)
		print(sess.run(h1, feed_dict={X: batch_x}))
		print(batch_y)