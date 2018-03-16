from math import *
import os
import argparse
from copy import copy

import sys
sys.path.insert(0, '../../../../')
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

class network(object):
	def __init__(self, params):
		# tf Graph input
		self.X = tf.placeholder(shape=[None, params['dof']], dtype=tf.float64, name="X")
		self.Y = tf.placeholder(shape=[None, params['numOutput']], dtype=tf.float64, name="Y")
		self.generateNetwork(params)

	def generateNetwork(self, params):

		self.joiningMatrix = (np.array([[1., -1., 0.]]))#0.5, -0.5, 0]]))
		
		for i in range(params['dof']-1):
			
			l = self.joiningMatrix.shape[1]
			self.combinationMatrix = np.vstack((1.0*np.ones(l), self.joiningMatrix))
			self.combinationMatrix = np.hstack((self.combinationMatrix,np.vstack((-1.0*np.ones(l), self.joiningMatrix))))
			# combinationMatrix = np.hstack((combinationMatrix,np.vstack((0.5*np.ones(l), joiningMatrix))))
			# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-0.5*np.ones(l), joiningMatrix))))
			self.combinationMatrix = np.hstack((self.combinationMatrix,np.vstack((1.0*np.zeros(l), self.joiningMatrix))))
			self.joiningMatrix = copy(self.combinationMatrix)
		
		self.combinationMatrix = np.delete(self.combinationMatrix,-1,1)
		print(self.combinationMatrix.shape)

		# Construct model
		# h_jointAngles = layers.fully_connected(inputs=X, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.sin) # Change sigmoid (relu, cos/sin)   |    is there bias ??
		self.inputCombinations = tf.matmul(self.X, self.combinationMatrix)
		self.h_jointAngles = tf.concat([tf.sin(self.inputCombinations), tf.cos(self.inputCombinations)],1)
		# self.h_jointAngles = tf.concat([X, self.h_jointAngles],1)
		self.hiddenLayer = layers.fully_connected(inputs=self.h_jointAngles, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		self.hiddenLayer = layers.fully_connected(inputs=self.hiddenLayer, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		self.hiddenLayer = layers.fully_connected(inputs=self.hiddenLayer, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		# self.hiddenLayer = layers.fully_connected(inputs=self.hiddenLayer, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		# self.hiddenLayer = layers.fully_connected(inputs=self.hiddenLayer, num_outputs=params['hiddenSize'], biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
		self.outputLayer = layers.fully_connected(inputs=self.hiddenLayer, num_outputs=params['numOutput'], biases_initializer=tf.zeros_initializer(), activation_fn=None)

		# Define loss and optimizer
		self.errorVector = ((tf.reduce_sum((self.outputLayer - self.Y)**2, 1)))
		self.cost = tf.sqrt(tf.reduce_mean(self.errorVector))
		self.reward = 100.0/params['batchSize']*tf.reduce_sum(tf.cast((tf.sqrt(self.errorVector) < params['tolerance']), tf.float64))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=params['learningRate'])
		self.train_op = self.optimizer.minimize(self.cost)



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
			params['learningRate'] = 0.00001
			params['numSteps'] = 10000
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
			params['numSteps'] = 12000
			params['batchSize'] = 64
			params['hiddenSize'] = 256

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
		
		params['tolerance'] = 5*1e-3
		robot.numOutput = params['numOutput']

		params['displayStep'] = 500
		params['saveStep'] = 1000
		print(params)

		# Network Parameters
		# numInput = 2*(3**dof-1)
		# normalize the gradients using clipping
		# GRAD_CLIP = 100
		# local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		# gradients = tf.gradients(cost, local_vars)
		# var_norms = tf.global_norm(local_vars)
		# grads, grad_norms = tf.clip_by_global_norm(gradients, GRAD_CLIP)

		# # Apply local gradients to global network
		# global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		# train_op = optimizer.apply_gradients(zip(grads, global_vars))

		nn = network(params)

		tf.summary.scalar("cost", nn.cost)
		tf.summary.scalar("reward", nn.reward)
		# tf.summary.scalar("gradients", tf.reduce_max(gradients))
		nn.merged_summary_op = tf.summary.merge_all()

		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()

		# Add ops to save and restore all the variables.
		saver = tf.train.Saver(max_to_keep=1)
		
	# if(args.mode =='train'):
		current_timestamp = make_log_dir('')
		# Start training
		with tf.Session() as sess:
			# Run the initializer
			sess.run(init)
			save_path = saver.save(sess, './model/' +  args.robot + '_NN.ckpt')
			log_dir = './train_log/' + args.robot + '_NN_' + current_timestamp
			os.makedirs(log_dir)
			summary_writer = tf.summary.FileWriter('./train_log/' + args.robot + '_NN_' + current_timestamp, graph=tf.get_default_graph())

			for step in range(1, params['numSteps']+1):
				batch_x, batch_y = robot.generateFKData(params['batchSize'])
				# Run optimization op (backprop)
				sess.run(nn.train_op, feed_dict={nn.X: batch_x, nn.Y: batch_y})
				if step % params['displayStep'] == 0 and step >1000:
					# Calculatsave_path = saver.save(sess, './model/' +  args.robot + '_NN/model-final'e batch loss and accuracy
					loss, summary = sess.run([nn.cost, nn.merged_summary_op], feed_dict={nn.X: batch_x, nn.Y: batch_y})
					print("step: {}, value: {}".format(step, loss))
					summary_writer.add_summary(summary, step)

				if step % params['saveStep'] == 0 or step == 1:
					# Save the variables to disk.
					save_path = saver.save(sess, './model/' +  args.robot + '_NN.ckpt', write_meta_graph=False)
			
			save_path = saver.save(sess, './model/' +  args.robot + '_NN.ckpt', write_meta_graph=False)
			print("Model saved in path: %s" % save_path)
			endLearningTime = time.time()
			results['trainingTime'] = endLearningTime - startTime
			# Check the values of the variables
			testBatchX, testBatchY = robot.generateFKTrajectory()
			y_pred = sess.run(nn.outputLayer, feed_dict={nn.X: testBatchX})

		results['errorVector'] = np.sqrt(np.sum((y_pred - testBatchY)**2, 1))
		results['averageCost'] = np.mean(results['errorVector'])
		results['reward'] = 100.0/robot.testSize*np.sum(results['errorVector'] < params['tolerance'])
		
		filename = './model/' + args.robot +'_NN_pickleData'
		pickle.dump([params, robot, results], open(filename, 'wb'))

	if(args.mode == 'test'):
		if args.model == None:
			filename = 'model/' + args.robot + '_NN_pickleData'
			modelName = 'model/' + args.robot + '_NN.ckpt.meta'
		params, robot, results = pickle.load(open(filename, 'rb'))	
		nn = network(params)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver(max_to_keep=1)

		with tf.Session() as sess:
			sess.run(init)
			saver = tf.train.import_meta_graph(modelName)
			savePath = './model/' #+ args.model + '_NN/'
			saver.restore(sess, tf.train.latest_checkpoint(savePath))
			print("Model restored.")
			
			# Check the values of the variables
			testBatchX, testBatchY = robot.generateFKTrajectory()
			y_pred = sess.run(nn.outputLayer, feed_dict={nn.X: testBatchX})

		results['errorVector'] = (np.sum((y_pred - testBatchY)**2, 1))
		results['averageCost'] = np.sqrt(np.mean(results['errorVector']))
		results['reward'] = 100.0/robot.testSize*np.sum(np.sqrt(results['errorVector']) < params['tolerance'])

	print("The total time taken for learning is :{} seconds".format(results['trainingTime']))
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
	plt.hist(results['errorVector'],np.linspace(0.0,0.01, 100))

	plt.show()

	print("Optimization Finished!")