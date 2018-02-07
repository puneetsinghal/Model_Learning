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

# Parameters
learningRate = 0.02
numSteps = 100000
batchSize = 100
displayStep = 100

# Network Parameters
numInput = 2
numOutput = 2

# tf Graph input
X = tf.placeholder(shape=[None, numInput], dtype=tf.float64)
Y = tf.placeholder(shape=[None, numOutput], dtype=tf.float64)

# Construct model
h0 = layers.fully_connected(inputs=X, num_outputs=100, activation_fn=tf.nn.sigmoid)
h1 = layers.fully_connected(inputs=h0, num_outputs=numOutput, activation_fn=None)

# Define loss and optimizer
cost = 1000*0.5*tf.reduce_mean((h1 - Y)**2)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRate)
train_op = optimizer.minimize(cost)

tf.summary.scalar("cost", cost)
merged_summary_op = tf.summary.merge_all()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)
	currentPath = os.getcwd()
	save_dir = currentPath + '/model/'
	summary_writer = tf.summary.FileWriter(save_dir, graph=tf.get_default_graph())

	for step in range(1, numSteps+1):
		batch_x, batch_y = generateData(batchSize)
		# Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
		if step % displayStep == 0 or step == 1:
			# Calculate batch loss and accuracy
			loss, summary = sess.run([cost, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
			print("step: {}, value: {}".format(step, loss))
			summary_writer.add_summary(summary, step)

	# Save the variables to disk.
	save_path = saver.save(sess, "./model/checkpoint.ckpt")
	print("Model saved in path: %s" % save_path)

print("Optimization Finished!")