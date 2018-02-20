

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
	learningRate = 0.001
	numSteps = 10000
	batchSize = 32
	hiddenSize = 100
	displayStep = 100
	saveStep = 10000

	# Network Parameters
	numInput = 2
	numOutput = 2

	# tf Graph input
	X = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
	L = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
	Y = tf.placeholder(shape=[None, numOutput], dtype=tf.float64)
	#L = tf.constant([[0.2, 0.2]], dtype=tf.float64)

	# Construct model
	h_jointAngles = layers.fully_connected(inputs=X, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=None) # Change sigmoid (relu, cos/sin)   |    is there bias ??
	h_lengths = layers.fully_connected(inputs=L, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=None)
	concateInput = tf.concat([h_jointAngles, h_lengths], 1)
	layer1 = layers.fully_connected(inputs=concateInput, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer2 = layers.fully_connected(inputs=layer1, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer3 = layers.fully_connected(inputs=layer2, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer4 = layers.fully_connected(inputs=layer3, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer5 = layers.fully_connected(inputs=layer4, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer6 = layers.fully_connected(inputs=layer5, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer7 = layers.fully_connected(inputs=layer6, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	outputLayer = layers.fully_connected(inputs=layer7, num_outputs=numOutput, biases_initializer=tf.zeros_initializer(), activation_fn=None)

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
		error = np.sum((out - batch_y)**2,1)
		print(error)
		print(np.sum(error)/batchSize)
		print(tf.reduce_mean((out - batch_y)**2))