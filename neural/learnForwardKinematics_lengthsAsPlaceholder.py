from math import *
import os
import argparse


import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt


def FK(lengths, angles):
	l1 = lengths[0]
	l2 = lengths[1]
	# l3 = 0.2
	# x = l1*cos(angles[0]) + l2*cos(angles[0] + angles[1]) + l3*cos(angles[0] + angles[1] + angles[2])
	# y = l1*sin(angles[0]) + l2*sin(angles[0] + angles[1]) + l3*sin(angles[0] + angles[1] + angles[2])
	x = l1*cos(angles[0]) + l2*cos(angles[0] + angles[1])
	y = l1*sin(angles[0]) + l2*sin(angles[0] + angles[1])
	return np.array([x, y])

def generateData(batchSize, dof=3):
	batch_x = pi*np.random.rand(batchSize, dof)
	batch_y = np.zeros((batchSize, 2))
	batch_l = np.ones((batchSize,dof)) * 0.2
	# batch_l = np.random.uniform(low=0.1, high=0.4, size=[batchSize, dof])
	for i in range(batchSize):
		batch_y[i, :] = FK(batch_l[i,:], batch_x[i,:])

	return batch_x, batch_y, batch_l

def generateTrajectory(dof):
	theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
	# theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
	theta_2 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
	batch_x = np.vstack((theta_1, theta_2))
	# batch_x = np.vstack((batch_x, theta_3))
	batch_x = batch_x.T
	batch_y = np.zeros((batch_x.shape[0],dof))
	batch_l = np.ones(batch_x.shape) * 0.2
	# batch_l = np.random.uniform(low=0.1, high=0.4, size=[batch_x.shape[0], dof])
	
	for i in range(batch_x.shape[0]):
		batch_y[i, :] = FK(batch_l[i,:],batch_x[i,:])

	return batch_x, batch_y, batch_l

def make_log_dir(log_parent_dir):
	import datetime, os
	current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	log_dir = os.path.join(log_parent_dir, current_timestamp)
	os.makedirs(log_dir)
	# create empty logfiles now
	# log_files = {
	#                   'train_loss': os.path.join(lod_dir, 'train_loss.txt'),
	#                   'train_episode_reward': os.path.join(lod_dir, 'train_episode_reward.txt'),
	#                   'test_episode_reward': os.path.join(lod_dir, 'test_episode_reward.txt')
	#                 }
	# for key in self.log_files:
	#   open(os.path.join(self.log_dir, self.log_files[key]), 'a').close()
	return log_dir

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='test')

	args = parser.parse_args()

	# Parameters
	dof = 2
	learningRate = 0.0005
	numSteps = 6000
	batchSize = 64
	hiddenSize = 256
	displayStep = 100
	saveStep = 10000

	# Network Parameters
	numInput = dof
	numOutput = 2

	log_parent_dir = './train_log'
	log_dir=''

	# tf Graph input
	X = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
	# L = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
	Y = tf.placeholder(shape=[None, numOutput], dtype=tf.float64)
	#L = tf.constant([[0.2, 0.2]], dtype=tf.float64)

	# Construct model
	h_jointAngles = layers.fully_connected(inputs=X, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.sin) # Change sigmoid (relu, cos/sin)   |    is there bias ??
	# h_lengths = layers.fully_connected(inputs=L, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=None)
	# concateInput = tf.concat([h_jointAngles, h_lengths], 1)
	layer1 = layers.fully_connected(inputs=h_jointAngles, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer2 = layers.fully_connected(inputs=layer1, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer3 = layers.fully_connected(inputs=layer2, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	# layer4 = layers.fully_connected(inputs=layer3, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	# layer5 = layers.fully_connected(inputs=layer4, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	# layer6 = layers.fully_connected(inputs=layer5, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	# layer7 = layers.fully_connected(inputs=layer6, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	outputLayer = layers.fully_connected(inputs=layer3, num_outputs=numOutput, biases_initializer=tf.zeros_initializer(), activation_fn=None)

	# Define loss and optimizer
	cost = 1000*0.5*tf.reduce_mean((outputLayer - Y)**2)
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
	train_op = optimizer.minimize(cost)

	tf.summary.scalar("cost", cost)
	merged_summary_op = tf.summary.merge_all()

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	# currentPath = os.getcwd()
	# log_dir = currentPath + '/train/'

	if(args.mode=='train'):
		log_dir = make_log_dir(log_parent_dir)
		# Start training
		with tf.Session() as sess:
			# Run the initializer
			sess.run(init)

			summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

			for step in range(1, numSteps+1):
				batch_x, batch_y, batch_l = generateData(batchSize, dof)
				# Run optimization op (backprop)
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})#, L: batch_l})
				if step % displayStep == 0 or step == 1:
					# Calculate batch loss and accuracy
					loss, summary = sess.run([cost, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})#, L:batch_l})
					print("step: {}, value: {}".format(step, loss))
					summary_writer.add_summary(summary, step)

				if step % saveStep == 0 or step == 1:
					# Save the variables to disk.
					save_path = saver.save(sess, "./model/checkpoint.ckpt")
			
			save_path = saver.save(sess, "./model/checkpoint.ckpt")
			print("Model saved in path: %s" % save_path)

	# elif(args.mode == 'test'):
	batchSize = 100
	with tf.Session() as sess:

		sess.run(init)
		# Restore variables from disk.
		saver = tf.train.import_meta_graph('./model/checkpoint.ckpt.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./model/'))
		print("Model restored.")
		# Check the values of the variables
		batch_x, batch_y, batch_l = generateTrajectory(dof)
		out = sess.run(outputLayer, feed_dict={X: batch_x})#, L:batch_l})
		# print(out.shape)
		# print(batch_y.shape)
		error = np.sum((out - batch_y)**2,1)
		# print(error)
		print("Error = {}".format(np.sum(error)/batchSize))
		print(tf.reduce_mean((out - batch_y)**2))

		plt.figure(1)
		plt.plot(batch_y[:,0], batch_y[:,1],'g*')
		plt.plot(out[:,0], out[:,1],'r*')
		plt.show()
	print("Optimization Finished!")