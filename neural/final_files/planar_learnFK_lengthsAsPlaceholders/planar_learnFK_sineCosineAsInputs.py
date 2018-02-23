from math import *
import os
import argparse
from copy import copy


import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt


def FK(lengths, angles):
	# RRR planar manipulator
	x = lengths[0]*cos(angles[0]) + lengths[1]*cos(angles[0] + angles[1]) + lengths[2]*cos(angles[0] + angles[1] + angles[2])
	y = lengths[0]*sin(angles[0]) + lengths[1]*sin(angles[0] + angles[1]) + lengths[2]*sin(angles[0] + angles[1] + angles[2])
	
	# RR planar manipulator
	# x = lengths[0]*cos(angles[0]) + lengths[1]*cos(angles[0] + angles[1])
	# y = lengths[0]*sin(angles[0]) + lengths[1]*sin(angles[0] + angles[1])

	# return output
	return np.array([x, y])

def generateData(batchSize, dof=3):
	batch_x = pi*np.random.rand(batchSize, dof)
	batch_y = np.zeros((batchSize, 2))
	batch_l = np.ones((batchSize, dof)) * 1.
	for i in range(batchSize):
		batch_y[i, :] = FK(batch_l[i,:], batch_x[i,:])

	return batch_x, batch_y, batch_l

def generateTrajectory(dof):
	theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
	theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
	theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
	batch_x = np.vstack((theta_1, theta_2))
	batch_x = np.vstack((batch_x, theta_3))
	batch_x = batch_x.T
	batch_y = np.zeros((batch_x.shape[0],2))
	batch_l = np.ones(batch_x.shape) * 1.
	
	for i in range(batch_x.shape[0]):
		batch_y[i, :] = FK(batch_l[i,:],batch_x[i,:])

	return batch_x, batch_y, batch_l

def make_log_dir(log_parent_dir):
	import datetime, os
	current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	log_dir = os.path.join('./train_log', current_timestamp)
	os.makedirs(log_dir)
	return current_timestamp

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='test')
	parser.add_argument('--model', type=str, default=None)

	args = parser.parse_args()

	# Parameters
	dof = 3
	learningRate = 0.0001
	numSteps = 12000
	batchSize = 64
	displayStep = 100
	saveStep = 10000
	tolerance = 5*1e-3

	# Network Parameters
	numInput = 2*(3**dof-1)
	numEncoders = 8
	numOutput = 2

	joiningMatrix = (np.array([[1, -1, 0]]))
	l = joiningMatrix.shape[1]
	combinationMatrix = np.vstack((np.ones(l), joiningMatrix))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1*np.ones(l), joiningMatrix))))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((np.zeros(l), joiningMatrix))))
	joiningMatrix = copy(combinationMatrix)
	l = joiningMatrix.shape[1]
	combinationMatrix = np.vstack((np.ones(l), joiningMatrix))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1*np.ones(l), joiningMatrix))))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((np.zeros(l), joiningMatrix))))
	combinationMatrix = np.delete(combinationMatrix,-1,1)

	# tf Graph input
	X = tf.placeholder(shape=[None, dof], dtype=tf.float64)
	Y = tf.placeholder(shape=[None, numOutput], dtype=tf.float64)
	
	# Construct model
	# h_jointAngles = layers.fully_connected(inputs=X, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.sin) # Change sigmoid (relu, cos/sin)   |    is there bias ??
	inputCombinations = tf.matmul(X, combinationMatrix)
	h_jointAngles = tf.concat([tf.sin(inputCombinations), tf.cos(inputCombinations)],1)
	layer1 = layers.fully_connected(inputs=h_jointAngles, num_outputs=256, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer2 = layers.fully_connected(inputs=layer1, num_outputs=256, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer3 = layers.fully_connected(inputs=layer2, num_outputs=256, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	# layer4 = layers.fully_connected(inputs=layer3, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu6)
	outputLayer = layers.fully_connected(inputs=layer3, num_outputs=numOutput, biases_initializer=tf.zeros_initializer(), activation_fn=None)

	# Define loss and optimizer
	errorVector = (tf.reduce_sum((outputLayer - Y)**2, 1))
	cost = 1e6*tf.reduce_mean(errorVector)
	reward = 100.0/batchSize*tf.reduce_sum(tf.cast((tf.sqrt(errorVector) < tolerance), tf.float64))
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
	train_op = optimizer.minimize(cost)

	tf.summary.scalar("cost", cost)
	tf.summary.scalar("reward", reward)
	merged_summary_op = tf.summary.merge_all()

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()
	
	if(args.mode=='train'):
		current_timestamp = make_log_dir('')
		# Start training
		with tf.Session() as sess:
			# Run the initializer
			sess.run(init)

			summary_writer = tf.summary.FileWriter('./train_log/' + current_timestamp, graph=tf.get_default_graph())

			for step in range(1, numSteps+1):
				batch_x, batch_y, batch_l = generateData(batchSize, dof)
				# Run optimization op (backprop)
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
				if step % displayStep == 0 or step == 1:
					# Calculate batch loss and accuracy
					loss, summary = sess.run([cost, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
					print("step: {}, value: {}".format(step, loss))
					summary_writer.add_summary(summary, step)

				if step % saveStep == 0 or step == 1:
					# Save the variables to disk.
					save_path = saver.save(sess, './model/' + 'model-' + current_timestamp + '.ckpt')
			
			save_path = saver.save(sess, './model/' +  'model-' + current_timestamp + '.ckpt')
			print("Model saved in path: %s" % save_path)

	batchSize = 700
	with tf.Session() as sess:
		sess.run(init)

		# Restore variables from disk.
		if(args.mode == 'train'):
			saver = tf.train.import_meta_graph( './model/' + 'model-' + current_timestamp + '.ckpt.meta')
			saver.restore(sess, tf.train.latest_checkpoint('./model/'))
		else:
			if not (args.model==None):
				saver = tf.train.import_meta_graph(args.model)
			saver.restore(sess, tf.train.latest_checkpoint('./model/'))

		print("Model restored.")

		# Check the values of the variables
		batch_x, batch_y, batch_l = generateTrajectory(dof)
		out = sess.run(outputLayer, feed_dict={X: batch_x})
		print(out.shape)
		errorVector = np.sqrt(np.sum((out - batch_y)**2, 1))
		cost = 0.5*np.mean(errorVector)
		reward = 100.0/batchSize*np.sum(errorVector < tolerance)
		print("Error = {}".format(cost))
		print("Success rate = {}".format(reward))

		plt.figure(1)
		plt.plot(batch_y[:,0], batch_y[:,1],'g*')
		plt.plot(out[:,0], out[:,1],'r*')
		plt.show()

		plt.figure(2)
		plt.plot(np.linspace(1, errorVector.size, errorVector.size), errorVector)
		plt.show()
	print("Optimization Finished!")