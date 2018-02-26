from math import *
import os
import argparse
from copy import copy


import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

class Robot(object):
	"""docstring for ClassName"""
	def __init__(self, dof, dh_parameters=None):
		if (dh_parameters.any()):
			# if dh_parameters.shape[1] != 4 or dh_parameters.shape[0] !=5:
				# error('Invalid DH_parametrs: Should be a matrix of size, 5 X 4')
			self.dof = dh_parameters.shape[0]
			self.imp_dof = dof
			self.link_length = (dh_parameters[:, 0]).reshape([1,self.dof])
			self.link_twist = (dh_parameters[:, 1]).reshape([1,self.dof])
			self.link_offset = (dh_parameters[:, 2]).reshape([1,self.dof])
			self.joint_angle = (dh_parameters[:, 3]).reshape([1,self.dof])

	def FK(lengths, angles):
		# RRR 3D manipulator
		r = lengths[1]*cos(angles[1]) + lengths[2]*cos(angles[1] + angles[2])
		z = lengths[1]*sin(angles[1]) + lengths[2]*sin(angles[1] + angles[2])
		
		x = r*cos(angles[0])
		y = r*sin(angles[0])
		endEffector = np.array([x, y, z])
		return endEffector

	def dhMatrix(self, index, jointAngles):
		theta = jointAngles[index]+ self.joint_angle[0, index]
		twist = self.link_twist[0, index]
		offset = self.link_offset[0, index]
		linkLength = self.link_length[0, index]

		return np.array([[cos(theta), -sin(theta)*cos(twist), sin(theta)*sin(twist), linkLength*cos(theta)],
						[sin(theta), cos(theta)*cos(twist), -cos(theta)*sin(twist), linkLength*sin(theta)],
						[0, sin(twist), cos(twist), offset],
						[0, 0, 0, 1]])

	def forwardKinematics(self, lengths, jointAngles):
		frames = np.zeros((4, 4, self.dof))

		for index in range(self.dof):

			transform = self.dhMatrix(index, jointAngles)
			if index != 0:
				frames[:, :, index] = np.matmul(frames[:, :, index - 1], transform)
			else:
				frames[:, :, index] = transform

		endEffector = frames[0:3, 3, self.dof-1]
		return endEffector

	def generateData(self, batchSize, numOutput):
		batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		# batch_x[:,3] = -(batch_x[:,1] - batch_x[:,2]) - pi/2
		batch_y = np.zeros((batchSize, numOutput))
		# batch_l = np.ones((batchSize, self.dof)) * 0.5
		batch_l = np.repeat(self.link_length, batchSize, 0)
		for i in range(batchSize):
			endEffector = self.forwardKinematics(batch_l[i,:], batch_x[i,:])
			batch_y[i, :] = endEffector
			# print(endEffector)


		return batch_x, batch_y, batch_l

	def generateTrajectory(self):
		num = 4
		# theta_1 = np.hstack((np.linspace(0,0,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(pi,pi,700)))
		theta_1 = np.hstack((np.linspace(0,0,700), np.linspace(1.,1.,700), np.linspace(2.,2.,700), np.linspace(pi,pi,700)))
		theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		
		theta_4 = np.linspace(0.2,0.2,700)
		# theta_4 = -(theta_2-theta_3) - pi/2
		theta_5 = np.linspace(0.2,0.2,700)
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,num,1)))
		batch_x = np.vstack((batch_x, np.repeat(theta_4,num,0)))
		batch_x = np.vstack((batch_x, np.repeat(theta_5,num,0)))
		batch_x = batch_x.T
		# batch_x = batch_x[0:700,:]
		# batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros([batch_x.shape[0], self.numOutput])
		# batch_l = np.ones(batch_x.shape) * 0.5
		batch_l = np.repeat(self.link_length, 700*num, 0)
		# print(batch_l.shape)
		for i in range(batch_x.shape[0]):
			endEffector = self.forwardKinematics(batch_l[i,:], batch_x[i,:])
			batch_y[i, :] = endEffector
			
		return batch_x, batch_y, batch_l

def make_log_dir(log_parent_dir):
	import datetime, os
	current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	log_dir = os.path.join('./train_log', current_timestamp)
	os.makedirs(log_dir)
	return current_timestamp

if __name__ == '__main__':
	startTime = time.clock()

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type=str, default='test')
	parser.add_argument('--model', type=str, default=None)

	args = parser.parse_args()

	# Parameters
	dof = 5
	learningRate = 0.0005
	numSteps = 50000
	batchSize = 128
	hiddenSize = 256
	displayStep = 100
	saveStep = 1000
	tolerance = 5*1e-3

	# Network Parameters
	numInput = 2*(3**dof-1)
	numEncoders = 8
	numOutput = 3

	joiningMatrix = (np.array([[1, -1, 0]]))#0.5, -0.5, 0]]))
	
	l = joiningMatrix.shape[1]
	combinationMatrix = np.vstack((1.0*np.ones(l), joiningMatrix))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1.0*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((0.5*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-0.5*np.ones(l), joiningMatrix))))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((1.0*np.zeros(l), joiningMatrix))))
	joiningMatrix = copy(combinationMatrix)
	
	l = joiningMatrix.shape[1]
	combinationMatrix = np.vstack((1.0*np.ones(l), joiningMatrix))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1.0*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((0.5*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-0.5*np.ones(l), joiningMatrix))))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((1.0*np.zeros(l), joiningMatrix))))
	joiningMatrix = copy(combinationMatrix)

	l = joiningMatrix.shape[1]
	combinationMatrix = np.vstack((1.0*np.ones(l), joiningMatrix))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1.0*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((0.5*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-0.5*np.ones(l), joiningMatrix))))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((1.0*np.zeros(l), joiningMatrix))))
	joiningMatrix = copy(combinationMatrix)
	
	l = joiningMatrix.shape[1]
	combinationMatrix = np.vstack((1.0*np.ones(l), joiningMatrix))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((-1.0*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((0.5*np.ones(l), joiningMatrix))))
	# combinationMatrix = np.hstack((combinationMatrix,np.vstack((-0.5*np.ones(l), joiningMatrix))))
	combinationMatrix = np.hstack((combinationMatrix,np.vstack((1.0*np.zeros(l), joiningMatrix))))
	joiningMatrix = copy(combinationMatrix)
	
	combinationMatrix = np.delete(combinationMatrix,-1,1)

	# tf Graph input
	X = tf.placeholder(shape=[None, dof], dtype=tf.float64)
	Y = tf.placeholder(shape=[None, numOutput], dtype=tf.float64)
	
	# Construct model
	# h_jointAngles = layers.fully_connected(inputs=X, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.sin) # Change sigmoid (relu, cos/sin)   |    is there bias ??
	inputCombinations = tf.matmul(X, combinationMatrix)
	h_jointAngles = tf.concat([tf.sin(inputCombinations), tf.cos(inputCombinations)],1)
	layer1 = layers.fully_connected(inputs=h_jointAngles, num_outputs=512, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer2 = layers.fully_connected(inputs=layer1, num_outputs=512, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer3 = layers.fully_connected(inputs=layer2, num_outputs=512, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	layer4 = layers.fully_connected(inputs=layer3, num_outputs=512, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
	# layer5 = layers.fully_connected(inputs=layer4, num_outputs=hiddenSize, biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu)
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

	d1 = 0.14
	d4 = 0.11
	d5 = 0.075 + 0.15
	dh_parameters = np.array([[0, pi/2, d1, 0],
						    [0.334, pi, 0, 0],
						    [0.334, pi, 0, 0],
						    [0, pi/2, d4, pi/2],
						    [0, 0, d5, 0]])
	robot = Robot(dof, dh_parameters)
	robot.numOutput = numOutput
	# batch_x, batch_y, batch_l = robot.generateTrajectory(numOutput)
	# plt.figure(1)
	# plt.axes(projection='3d')
	# plt.plot(batch_y[:,0], batch_y[:,1], batch_y[:,2], 'g*')
	# plt.show()
	
	if(args.mode=='train'):
		current_timestamp = make_log_dir('')
		# Start training
		with tf.Session() as sess:
			# Run the initializer
			sess.run(init)

			summary_writer = tf.summary.FileWriter('./train_log/' + current_timestamp, graph=tf.get_default_graph())

			for step in range(1, numSteps+1):
				batch_x, batch_y, batch_l = robot.generateData(batchSize, numOutput)
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

	batchSize = 4*700
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
		batch_x, batch_y, batch_l = generateTrajectory()
		out = sess.run(outputLayer, feed_dict={X: batch_x})
		print(out.shape)
		errorVector = np.sqrt(np.sum((out - batch_y)**2, 1))
		cost = 0.5*np.mean(errorVector)
		reward = 100.0/batchSize*np.sum(errorVector < tolerance)
		print("Error = {}".format(cost))
		print("Success rate = {}".format(reward))
		print(errorVector)
		fig = plt.figure(1)
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(batch_y[:,0], batch_y[:,1], batch_y[:,2], 'g*')
		ax.scatter(out[:,0], out[:,1], out[:,2], 'r*')
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')

		plt.figure(2)
		plt.plot(np.linspace(1, errorVector.size, errorVector.size), errorVector)
		plt.show()
	endTime = time.clock()

	print("The total time taken is :{} seconds".format((endTime-startTime)))
	print("Optimization Finished!")