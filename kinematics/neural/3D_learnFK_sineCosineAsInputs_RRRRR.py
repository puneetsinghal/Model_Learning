from math import *
import os
import argparse
from copy import copy

import sys
sys.path.insert(0, '../../')
from robot import PlanarRR
from robot import PlanarRRR
from robot import NonPlanarRRR
from robot import NonPlanarRRRR
from robot import NonPlanarRRRRR


import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle


def make_log_dir(log_parent_dir):
	import datetime, os
	current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	return current_timestamp

if __name__ == '__main__':
	startTime = time.time()
	np.random.seed(1)
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='test')
	parser.add_argument('--robot', type=str, default=None)
	parser.add_argument('--model', type=str, default=None)

	args = parser.parse_args()

	if(args.mode == 'train'):
		# Parameters
		params = {}
		results = {}

		if(args.robot == "planarRR"):
			params['dof'] = 2
			params['numOutput'] = 2
			params['learningRate'] = 0.000001
			params['numSteps'] = 30000
			params['batchSize'] = 32
			params['hiddenSize'] = 256

			params['linkLengths'] = np.array([1., 1.])
			params['mLink'] = np.array([0.1, 0.1])
			params['inertia'] = np.array([0., 0.])
			params['gravity'] = -9.81
			robot = PlanarRR(params)

		if(args.robot == "planarRRR"):
			params['dof'] = 3
			params['numOutput'] = 2
			params['learningRate'] = 0.0005
			params['numSteps'] = 6000
			params['batchSize'] = 128
			params['hiddenSize'] = 128

			params['linkLengths'] = np.array([1., 1., 1.])
			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			robot = PlanarRRR(params)
			
		if(args.robot == "nonPlanarRRR"):
			params['dof'] = 3
			params['numOutput'] = 3
			params['learningRate'] = 0.0005
			params['numSteps'] = 6000
			params['batchSize'] = 128
			params['hiddenSize'] = 128

			params['linkLengths'] = np.array([1., 1., 1.])
			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			robot = NonPlanarRRR(params)
			
		if(args.robot == "nonPlanarRRRR"):
			params['dof'] = 5 # 5th joint is fixed
			params['numOutput'] = 3
			params['learningRate'] = 0.0005
			params['numSteps'] = 6000
			params['batchSize'] = 128
			params['hiddenSize'] = 128

			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			d1 = 0.14
			d4 = 0.11
			d5 = 0.075 + 0.15
			params['dh_parameters'] = np.array([[0, pi/2, d1, 0],
								    [0.334, pi, 0, 0],
								    [0.334, pi, 0, 0],
								    [0, pi/2, d4, pi/2],
								    [0, 0, d5, 0]])
			robot = NonPlanarRRRR(params)

		if(args.robot == "nonPlanarRRRRR"):
			params['dof'] = 5
			params['numOutput'] = 3
			params['learningRate'] = 0.0001
			params['numSteps'] = 50000
			params['batchSize'] = 128
			params['hiddenSize'] = 1024

			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			d1 = 0.14
			d4 = 0.11
			d5 = 0.075 + 0.15
			params['dh_parameters'] = np.array([[0, pi/2, d1, 0],
								    [0.334, pi, 0, 0],
								    [0.334, pi, 0, 0],
								    [0, pi/2, d4, pi/2],
								    [0, 0, d5, 0]])
			robot = NonPlanarRRRRR(params)
		
		params['tolerance'] = 1*1e-3
		robot.numOutput = params['numOutput']

		params['displayStep'] = 500
		params['saveStep'] = 1000
		print(params)

		# Network Parameters
		# numInput = 2*(3**dof-1)

		joiningMatrix = (np.array([[1, 0]]))#0.5, -0.5, 0]]))
		
		for i in range(params['dof']-1):
			
			l = joiningMatrix.shape[1]
			combinationMatrix = np.vstack((1.0*np.ones(l), joiningMatrix))
			# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1.0*np.ones(l), joiningMatrix))))
			# combinationMatrix = np.hstack((combinationMatrix,np.vstack((0.5*np.ones(l), joiningMatrix))))
			# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-0.5*np.ones(l), joiningMatrix))))
			combinationMatrix = np.hstack((combinationMatrix,np.vstack((1.0*np.zeros(l), joiningMatrix))))
			joiningMatrix = copy(combinationMatrix)
		
		combinationMatrix = np.delete(combinationMatrix,-1,1)
		print(combinationMatrix.shape)
		# tf Graph input
		X = tf.placeholder(shape=[None, params['dof']], dtype=tf.float64)
		Y = tf.placeholder(shape=[None, params['numOutput']], dtype=tf.float64)
		
		# Construct model
		# h_jointAngles = layers.fully_connected(inputs=X, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.sin) # Change sigmoid (relu, cos/sin)   |    is there bias ??
		inputCombinations = tf.matmul(X, combinationMatrix)
		h_jointAngles = tf.concat([tf.sin(inputCombinations), tf.cos(inputCombinations)],1)
		layer1 = layers.fully_connected(inputs=h_jointAngles, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		layer2 = layers.fully_connected(inputs=layer1, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		layer3 = layers.fully_connected(inputs=layer2, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		# layer4 = layers.fully_connected(inputs=layer3, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		# layer5 = layers.fully_connected(inputs=layer4, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		outputLayer = layers.fully_connected(inputs=layer3, num_outputs=params['numOutput'], biases_initializer=tf.zeros_initializer(), activation_fn=None)

		# Define loss and optimizer
		errorVector = ((tf.reduce_sum((outputLayer - Y)**2, 1)))
		cost = 1e6*tf.reduce_mean(errorVector)
		reward = 100.0/params['batchSize']*tf.reduce_sum(tf.cast((tf.sqrt(errorVector) < params['tolerance']), tf.float64))
		optimizer = tf.train.AdamOptimizer(learning_rate=params['learningRate'])
		train_op = optimizer.minimize(cost)

		tf.summary.scalar("cost", cost)
		tf.summary.scalar("reward", reward)
		merged_summary_op = tf.summary.merge_all()

		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()

		# Add ops to save and restore all the variables.
		saver = tf.train.Saver()
		
	# if(args.mode=='train'):
		current_timestamp = make_log_dir('')
		# Start training
		with tf.Session() as sess:
			# Run the initializer
			sess.run(init)
			log_dir = './train_log/' + args.robot + '_NN_' + current_timestamp
			os.makedirs(log_dir)
			summary_writer = tf.summary.FileWriter('./train_log/' + args.robot + '_NN_' + current_timestamp, graph=tf.get_default_graph())

			for step in range(1, params['numSteps']+1):
				batch_x, batch_y = robot.generateFKData(params['batchSize'])
				# Run optimization op (backprop)
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
				if step % params['displayStep'] == 0 or step == 1:
					# Calculate batch loss and accuracy
					loss, summary = sess.run([cost, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
					print("step: {}, value: {}".format(step, loss))
					summary_writer.add_summary(summary, step)

				if step % params['saveStep'] == 0 or step == 1:
					# Save the variables to disk.
					save_path = saver.save(sess, './model/' +  args.robot + '_NN_' + str(step) + '.ckpt')
			
			save_path = saver.save(sess, './model/' +  args.robot + '_NN_' + str(step) + '.ckpt')
			print("Model saved in path: %s" % save_path)
			endLearningTime = time.time()
			results['trainingTime'] = endLearningTime - startTime
			# Check the values of the variables
			testBatchX, testBatchY = robot.generateFKTrajectory()
			y_pred = sess.run(outputLayer, feed_dict={X: testBatchX})

		results['errorVector'] = np.sqrt(np.sum((y_pred - testBatchY)**2, 1))
		results['averageCost'] = np.mean(results['errorVector'])
		results['reward'] = 100.0/robot.testSize*np.sum(results['errorVector'] < params['tolerance'])
		
		filename = './train_log/' + args.robot +'_NN_' + current_timestamp + '_pickleData'
		pickle.dump([params, robot, results], open(filename, 'wb'))

	if(args.mode == 'test'):
		filename = args.model  + '_pickleData'
		params, robot, results = pickle.load(open(filename, 'rb'))	

		with tf.Session() as sess:
			sess.run(init)
			saver = tf.train.import_meta_graph(args.model)
			saver.restore(sess, tf.train.latest_checkpoint('./model/'))
			print("Model restored.")
	
			# Check the values of the variables
			testBatchX, testBatchY = robot.generateFKTrajectory()
			y_pred = sess.run(outputLayer, feed_dict={X: testBatchX})

		results['errorVector'] = np.sqrt(np.sum((y_pred - testBatchY)**2, 1))
		results['averageCost'] = np.mean(results['errorVector'])
		results['reward'] = 100.0/robot.testSize*np.sum(results['errorVector'] < params['tolerance'])


	print("Success rate = {}".format(results['reward']))
	print("Average error is:{}".format(results['averageCost']))

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	if(params['numOutput'] == 2):	
		ax.scatter(testBatchY[:,0], testBatchY[:,1], 0.0*testBatchY[:,1], 'g*')
		ax.scatter(y_pred[:,0], y_pred[:,1], 0.0*y_pred[:,1], 'r*')
	else:
		ax.scatter(testBatchY[:,0], testBatchY[:,1], testBatchY[:,2], 'g*')
		ax.scatter(y_pred[:,0], y_pred[:,1], y_pred[:,2], 'r*')

	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')

	plt.figure(2)
	plt.plot(np.linspace(1, results['errorVector'].size, results['errorVector'].size), results['errorVector'])

	plt.figure(3)
	plt.hist(results['errorVector'],np.linspace(0.00001,0.001, 100))

	plt.show()

	print("Optimization Finished!")