from math import *
import numpy as np

class PlanarRR(object):
	"""Planar RR manipulator"""
	def __init__(self, dof, linkLength=None):
		self.dof = dof
		self.link_length = linkLength 

	def forwardKinematics(self, angles):
		# Planar RR manipulator
		x = self.link_length[0]*cos(angles[0]) + self.link_length[1]*cos(angles[0] + angles[1])
		y = self.link_length[0]*sin(angles[0]) + self.link_length[1]*sin(angles[0] + angles[1])
		
		endEffector = np.array([x, y])
		return endEffector

	def generateData(self, batchSize):
		np.random.seed(1)
		batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros((batchSize, self.numOutput))
		
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_2 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		batch_x = np.vstack((theta_1, theta_2))
		batch_x = batch_x.T

		self.testSize = batch_x.shape[0]
		batch_y = np.zeros((self.testSize, self.numOutput))

		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])
			
		return batch_x, batch_y

class PlanarRRR(object):
	"""Planar RRR manipulator"""
	def __init__(self, dof, linkLength=None):
		self.dof = dof
		self.link_length = linkLength 

	def forwardKinematics(self, angles):
		x = self.link_length[0]*cos(angles[0]) + self.link_length[1]*cos(angles[0] + angles[1]) + self.link_length[2]*cos(angles[0] + angles[1] + angles[2])
		y = self.link_length[0]*sin(angles[0]) + self.link_length[1]*sin(angles[0] + angles[1]) + self.link_length[2]*sin(angles[0] + angles[1] + angles[2])
		
		endEffector = np.array([x, y])
		return endEffector

	def generateData(self, batchSize):
		np.random.seed(1)
		batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros((batchSize, self.numOutput))
		
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(pi,pi,700)))
		theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,7,1)))
		batch_x = batch_x.T

		self.testSize = batch_x.shape[0]
		batch_y = np.zeros((self.testSize, self.numOutput))

		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])
			
		return batch_x, batch_y


class NonPlanarRRR(object):
	"""Planar RRR manipulator"""
	def __init__(self, dof, linkLength=None):
		self.dof = dof
		self.link_length = linkLength 

	def forwardKinematics(self, angles):
		# RRR 3D manipulator
		r = self.link_length[1]*cos(angles[1]) + self.link_length[2]*cos(angles[1] + angles[2])
		z = self.link_length[1]*sin(angles[1]) + self.link_length[2]*sin(angles[1] + angles[2])
		
		x = r*cos(angles[0])
		y = r*sin(angles[0])
		endEffector = np.array([x, y, z])
		return endEffector

	def generateData(self, batchSize):
		np.random.seed(1)
		batch_x = pi*np.random.rand(batchSize, self.dof)
		batch_y = np.zeros((batchSize, self.numOutput))

		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(pi,pi,700)))
		theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,7,1)))
		batch_x = batch_x.T

		self.testSize = batch_x.shape[0]
		batch_y = np.zeros((self.testSize, self.numOutput))
		
		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

class NonPlanarRRRRR(object):
	"""docstring for ClassName"""
	def __init__(self, dof, dh_parameters=None):
		if (dh_parameters.any()):
			self.dof = dh_parameters.shape[0]
			self.imp_dof = dof
			self.link_length = (dh_parameters[:, 0]).reshape([1,self.dof])
			self.link_twist = (dh_parameters[:, 1]).reshape([1,self.dof])
			self.link_offset = (dh_parameters[:, 2]).reshape([1,self.dof])
			self.joint_angle = (dh_parameters[:, 3]).reshape([1,self.dof])

	def dhMatrix(self, index, jointAngles):
		theta = jointAngles[index]+ self.joint_angle[0, index]
		twist = self.link_twist[0, index]
		offset = self.link_offset[0, index]
		linkLength = self.link_length[0, index]

		return np.array([[cos(theta), -sin(theta)*cos(twist), sin(theta)*sin(twist), linkLength*cos(theta)],
						[sin(theta), cos(theta)*cos(twist), -cos(theta)*sin(twist), linkLength*sin(theta)],
						[0, sin(twist), cos(twist), offset],
						[0, 0, 0, 1]])

	def forwardKinematics(self, jointAngles):
		frames = np.zeros((4, 4, self.dof))

		for index in range(self.dof):
			transform = self.dhMatrix(index, jointAngles)
			if index != 0:
				frames[:, :, index] = np.matmul(frames[:, :, index - 1], transform)
			else:
				frames[:, :, index] = transform

		endEffector = frames[0:3, 3, self.dof-1]
		return endEffector

	def generateData(self, batchSize):
		batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros((batchSize, self.numOutput))
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(pi,pi,700)))
		theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		theta_4 = np.linspace(0.2,0.2,700)
		theta_5 = np.linspace(0.2,0.2,700)
		
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,7,1)))
		batch_x = np.vstack((batch_x, np.repeat(theta_4,7,0)))
		batch_x = np.vstack((batch_x, np.repeat(theta_5,7,0)))
		batch_x = batch_x.T
		
		self.testSize = batch_x.shape[0]
		batch_y = np.zeros([self.testSize, self.numOutput])
		
		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])
			
		return batch_x, batch_y