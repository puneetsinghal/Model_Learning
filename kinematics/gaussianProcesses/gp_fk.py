
from math import *
import os
import argparse
from copy import copy
import time

import sys
sys.path.insert(0, '../')
from robot import PlanarRR
from robot import PlanarRRR
from robot import NonPlanarRRR
from robot import NonPlanarRRRRR

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from IPython import embed
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata

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
	parser.add_argument('--robot', type=str, default=None)
	parser.add_argument('--model', type=str, default=None)

	args = parser.parse_args()

	# Parameters
	if(args.robot == "planarRR"):
		dof = 2
		numOutput = 2
		linkLengths = np.array([1., 1.])
		batchSize = 100
		robot = PlanarRR(dof, linkLengths)

	if(args.robot == "planarRRR"):
		dof = 3
		numOutput = 2
		linkLengths = np.array([1., 1., 1.])
		batchSize = 1000
		robot = PlanarRRR(dof, linkLengths)
		
	if(args.robot == "nonPlanarRR"):
		dof = 3
		numOutput = 3
		linkLengths = np.array([1., 1., 1.])
		batchSize = 1000
		robot = NonPlanarRRR(dof, linkLengths)
		
	if(args.robot == "planarRR"):
		dof = 5
		numOutput = 3
		batchSize = 5000
		d1 = 0.14
		d4 = 0.11
		d5 = 0.075 + 0.15
		dh_parameters = np.array([[0, pi/2, d1, 0],
							    [0.334, pi, 0, 0],
							    [0.334, pi, 0, 0],
							    [0, pi/2, d4, pi/2],
							    [0, 0, d5, 0]])
		robot = NonPlanarRRRRR(dof, dh_parameters)
	
	tolerance = 1*1e-4
	optimizationStep = 1
	robot.numOutput = numOutput

	X, Y = robot.generateData(batchSize)
	kernel = C(1.0, (1e-3, 1e3)) * RBF(12, (1e-2, 1e2))
	gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=optimizationStep)
	gp.fit(X, Y)

	endLearningTime = time.clock()
	print("The total time taken for learning is :{} seconds".format((endLearningTime-startTime)))


	testBatchX, testBatchY = robot.generateTrajectory()

	y_pred, sigma = gp.predict(testBatchX, return_std=True)
	endTestTime = time.clock()
	print("The total time taken for testing is :{} seconds".format((endTestTime-endLearningTime)))

	errorVector = np.sqrt(np.sum((y_pred - testBatchY)**2, 1))
	averageCost = np.mean(errorVector)
	reward = 100.0/robot.testSize*np.sum(errorVector < tolerance)

	print("Success rate = {}".format(reward))
	print("Average error is:{}".format(averageCost))

	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	if(robot.numOutput == 2):	
		ax.scatter(testBatchY[:,0], testBatchY[:,1], 0.0*testBatchY[:,1], 'g*')
		ax.scatter(y_pred[:,0], y_pred[:,1], 0.0*y_pred[:,1], 'r*')
	else:
		ax.scatter(testBatchY[:,0], testBatchY[:,1], testBatchY[:,2], 'g*')
		ax.scatter(y_pred[:,0], y_pred[:,1], y_pred[:,2], 'r*')

	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')

	plt.figure(2)
	plt.plot(np.linspace(1, errorVector.size, errorVector.size), errorVector)

	plt.figure(3)
	plt.hist(errorVector,np.linspace(0.00001,0.001, 100))

	plt.show()