
from math import *
import os
import argparse
from copy import copy
import time

import sys
sys.path.insert(0, '../../')
from robot import PlanarRR
from robot import PlanarRRR
from robot import NonPlanarRRR
from robot import NonPlanarRRRR
from robot import NonPlanarRRRRR

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from IPython import embed
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
import pickle

def make_log_dir(log_parent_dir):
	import datetime, os
	current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	# log_dir = os.path.join('./train_log', current_timestamp)
	# os.makedirs(log_dir)
	return current_timestamp

if __name__ == '__main__':

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
			params['linkLengths'] = np.array([1., 1.])
			params['mLink'] = np.array([0.1, 0.1])
			params['inertia'] = np.array([0., 0.])
			params['gravity'] = -9.81
			params['batchSize'] = 2000
			params['optimizationStep'] = 1
			robot = PlanarRR(params)

		if(args.robot == "planarRRR"):
			params['dof'] = 3
			params['numOutput'] = 2
			params['linkLengths'] = np.array([1., 1., 1.])
			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			params['batchSize'] = 3000
			params['optimizationStep'] = 1
			robot = PlanarRRR(params)
			
		if(args.robot == "nonPlanarRRR"):
			params['dof'] = 3
			params['numOutput'] = 3
			params['linkLengths'] = np.array([1., 1., 1.])
			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			params['batchSize'] = 1000
			params['optimizationStep'] = 1
			robot = NonPlanarRRR(params)
			
		if(args.robot == "nonPlanarRRRR"):
			params['dof'] = 5 # 5th joint is fixed
			params['numOutput'] = 3
			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			params['batchSize'] = 5000
			params['optimizationStep'] = 1
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
			params['mLink'] = None
			params['inertia'] = None
			params['gravity'] = -9.81
			params['batchSize'] = 10000
			params['optimizationStep'] = 1
			d1 = 0.14
			d4 = 0.11
			d5 = 0.075 + 0.15
			params['dh_parameters'] = np.array([[0, pi/2, d1, 0],
								    [0.334, pi, 0, 0],
								    [0.334, pi, 0, 0],
								    [0, pi/2, d4, pi/2],
								    [0, 0, d5, 0]])
			robot = NonPlanarRRRRR(params)
		
		params['tolerance'] = 1*1e-4
		print(params)
		robot.numOutput = params['numOutput']

		X, Y = robot.generateInverseDynamicsData(params['batchSize'])
		
		startTime = time.time()
		kernel = C(1.0, (1e-3, 1e3)) * RBF(12, (1e-2, 1e2))
		gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=params['optimizationStep'])
		gp.fit(X, Y)

		endLearningTime = time.time()
		results['trainingTime'] = endLearningTime - startTime
		
		testBatchX, testBatchY = robot.generateInverseDynamicsTrajectory()

		y_pred, sigma = gp.predict(testBatchX, return_std=True)
		endTestTime = time.time()
		results['testingTime'] = endTestTime - endLearningTime
		
		results['errorVector'] = np.sqrt(np.sum((y_pred - testBatchY)**2, 1))
		results['averageCost'] = np.mean(results['errorVector'])
		results['reward'] = 100.0/robot.testSize*np.sum(results['errorVector'] < params['tolerance'])

		filename = './train_log/' + args.robot +'_' + make_log_dir('./train_log/')
		pickle.dump([params, robot, gp, results], open(filename, 'wb'))

	if(args.mode == 'test'):
		startTime = time.time()
		# load the data [params, gp, results] from disk
		filename = args.model
		params, robot, gp, results = pickle.load(open(filename, 'rb'))
		params['tolerance'] = 1*1e-3
		robot.numOutput = params['numOutput']

		testBatchX, testBatchY = robot.generateInverseDynamicsTrajectory()

		y_pred, sigma = gp.predict(testBatchX, return_std=True)
		
		endTestTime = time.time()
		results['testingTime'] = endTestTime - startTime
		
		results['errorVector'] = np.sqrt(np.sum((y_pred - testBatchY)**2, 1))
		results['averageCost'] = np.mean(results['errorVector'])
		results['reward'] = 100.0/robot.testSize*np.sum(results['errorVector'] < params['tolerance'])

	print("The total time taken for learning is :{} seconds".format(results['trainingTime']))
	print("The total time taken for testing is :{} seconds".format(results['testingTime']))

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
