
from scipy.optimize import minimize
import numdifftools as nd
import numpy as np
from math import *
from copy import copy
from IPython import embed

class INVOPT(object):
	"""docstring for MPC"""
	def __init__(self, robot, params, nn, bnds, SESS):
		self.robot = robot
		self.sess = SESS
		self.params = params
		LB = -np.pi
		UB = np.pi
		self.bnds = ((LB, UB), (LB, UB))
		self.nn = nn

		# self.x = params['x0']
		# self.u0 = params['u0']
		# self.xref = params['x0']
		# self.cons = ({'type': 'ineq', 'fun': self.Contraints, 'args':()})

	def CostFunction(self, x):
	    # Cost function taken from MatLab's nMPC example code 
		J = 0.
		# embed()
		pos = self.sess.run(self.nn.outputLayer, feed_dict={self.nn.X: x.reshape(1,2)})

		J = 0.5*np.sum((pos - self.targetPos)**2)
		# embed()
		# print(J)
		return J

	def Jacobian(self, x):
		pos = self.sess.run(self.nn.outputLayer, feed_dict={self.nn.X: x.reshape(1,2)})

		g = np.zeros([self.params['numOutput'], self.params['dof']])
		if (self.params['numOutput'] == 2):
			g1, g2 = self.sess.run([self.nn.grad1, self.nn.grad2], 
					feed_dict={self.nn.X: x.reshape(1,2)})
			g[0,:] = g1[0]
			g[1,:] = g2[0]
		
		if (self.params['numOutput'] == 3):
			g1, g2, g3 = self.sess.run([self.nn.grad1, self.nn.grad2, self.nn.grad3], 
					feed_dict={self.nn.X: x.reshape(1,2)})
			g[0,:] = g1[0]
			g[1,:] = g2[0]
			g[2,:] = g2[0]

		if (self.params['numOutput'] == 4):
			g1, g2, g3, g4 = self.sess.run([self.nn.grad1, self.nn.grad2, self.nn.grad3, self.nn.grad4], 
					feed_dict={self.nn.X: x.reshape(1,2)})
			g[0,:] = g1[0]
			g[1,:] = g2[0]
			g[2,:] = g3[0]
			g[3,:] = g4[0]
		
		if (self.params['numOutput'] == 5):
			g1, g2, g3, g4, g5 = self.sess.run([self.nn.grad1, self.nn.grad2, self.nn.grad3, self.nn.grad4, self.nn.grad5], 
					feed_dict={self.nn.X: x.reshape(1,2)})
			g[0,:] = g1[0]
			g[1,:] = g2[0]
			g[2,:] = g3[0]
			g[3,:] = g4[0]
			g[4,:] = g5[0]
		
		if (self.params['numOutput'] == 6):
			g1, g2, g3, g4, g5, g6 = self.sess.run([self.nn.grad1, self.nn.grad2, self.nn.grad3, self.nn.grad4, self.nn.grad5, self.nn.grad6], 
					feed_dict={self.nn.X: x.reshape(1,2)})
			g[0,:] = g1[0]
			g[1,:] = g2[0]
			g[2,:] = g3[0]
			g[3,:] = g4[0]
			g[4,:] = g5[0]
			g[5,:] = g6[0]
		
		if (self.params['numOutput'] == 7):
			g1, g2, g3, g4, g5, g6, g7 = self.sess.run([self.nn.grad1, self.nn.grad2, self.nn.grad3, self.nn.grad4, self.nn.grad5, self.nn.grad6, self.nn.grad7], 
					feed_dict={self.nn.X: x.reshape(1,2)})
			g[0,:] = g1[0]
			g[1,:] = g2[0]
			g[2,:] = g3[0]
			g[3,:] = g4[0]
			g[4,:] = g5[0]
			g[5,:] = g6[0]
			g[6,:] = g7[0]
		
		# embed()
		jac = np.matmul((pos - self.targetPos), g).T
		# jac = np.matmul((pos - self.targetPos),np.vstack((g1[0], g2[0]))).T
		
		# print(jac)
		return jac
		

	def optimize(self, theta_initial, fk):
		self.targetPos = fk
		results = minimize(self.CostFunction, theta_initial, args=(), method='SLSQP', jac=self.Jacobian, bounds=self.bnds, 
			options={'disp': False, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 10000, 'ftol': 1e-10}) 

		# results = minimize(self.CostFunction, theta_initial, args=(), method='L-BFGS-B', jac=self.Jacobian, bounds=self.bnds, 
		# 	options={'disp': None, 'maxls': 100, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 
		# 	'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
		return results
