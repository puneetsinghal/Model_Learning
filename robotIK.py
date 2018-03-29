from math import *
import numpy as np

class PlanarRR(object):
	"""Planar RR manipulator"""
	def __init__(self, params):
		self.dof = params['dof']
		self.link_length = params['linkLengths']
		self.mLink = params['mLink']
		self.inertia = params['inertia']
		self.gravity = params['gravity']

	def forwardKinematics(self, angles):
		# Planar RR manipulator
		x = self.link_length[0]*cos(angles[0]) + self.link_length[1]*cos(angles[0] + angles[1])
		y = self.link_length[0]*sin(angles[0]) + self.link_length[1]*sin(angles[0] + angles[1])
		
		endEffector = np.array([x, y])
		return endEffector

	def inverseKinematics(self, position):
		# Planar RR manipulator
		# print(position.shape)
		cosineTheta2 = (position[0]**2 + position[1]**2 - self.link_length[0]**2 - self.link_length[1]**2)/(2*self.link_length[0]*self.link_length[1])
		sineTheta2 = pow((1 - cosineTheta2**2), 0.5)
		theta2 = atan2(sineTheta2, cosineTheta2)

		sineTheta1 = ((self.link_length[0] + self.link_length[1]*cosineTheta2)*position[1] - self.link_length[1]*sineTheta2*position[0])/(position[0]**2 + position[1]**2)
		cosineTheta1 = ((self.link_length[0] + self.link_length[1]*cosineTheta2)*position[0] + self.link_length[1]*sineTheta2*position[1])/(position[0]**2 + position[1]**2)
		theta1 = atan2(sineTheta1, cosineTheta1)
		
		theta = np.array([theta1, theta2])
		return theta

	def generateFKData(self, batchSize):
		# np.random.seed(1)
		batch_x = pi*np.random.rand(batchSize, self.dof) - 0.*pi*np.ones([batchSize, self.dof])
		# batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros((batchSize, self.numOutput))
		
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	# def generateIKData(self, batchSize):
	# 	# np.random.seed(1)
	# 	# batch_x = pi*np.random.rand(batchSize, self.dof) - 0.*pi*np.ones([batchSize, self.dof])
	# 	# # batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		
	# 	R = self.link_length[0] + self.link_length[1]
	# 	batch_y = np.zeros((batchSize, self.dof))
	# 	batch_radius = 0.5*R + 0.5*R*np.random.rand(batchSize, 1)
	# 	batch_theta = 2.0*pi*np.random.rand(batchSize, 1)
	# 	batch_x = np.hstack((batch_radius*np.cos(batch_theta), batch_radius*np.sin(batch_theta)))
	# 	# print(batch_x.shape)
	# 	for i in range(batchSize):
	# 		batch_y[i, :] = self.inverseKinematics(batch_x[i,:])

	# 	return batch_x, batch_y

	# def generateIKTrajectory(self):
	# 	# theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
	# 	# theta_2 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
	# 	# batch_x = np.vstack((theta_1, theta_2))
	# 	# batch_x = batch_x.T

	# 	R = self.link_length[0] + self.link_length[1]
	# 	batch_radius = np.linspace(0.5*R,R,700).reshape(700,1)
	# 	# print(radius.shape)
	# 	batch_theta = np.repeat(np.linspace(0.0,2.0*pi,100),7,0).reshape(700,1)
	# 	# print(theta.shape)
	# 	batch_x = np.hstack((batch_radius*np.cos(batch_theta), batch_radius*np.sin(batch_theta)))
	# 	# print(batch_x.shape)
	# 	# batch_x = np.vstack((radius, theta))
	# 	# batch_x = batch_x.T

	# 	self.testSize = batch_x.shape[0]
	# 	batch_y = np.zeros((self.testSize, self.dof))
	# 	# print(batch_x.shape)
	# 	# print(batch_y.shape)
	# 	for i in range(self.testSize):
	# 		batch_y[i, :] = self.inverseKinematics(batch_x[i,:])
			
	# 	return batch_x, batch_y

	def generateIKData(self, batchSize):
		batch_y, batch_x = self.generateFKData(batchSize)
		return batch_x, batch_y

	def generateIKTrajectory(self):
		batch_y, batch_x = self.generateFKTrajectory()
		return batch_x, batch_y

	def generateFKTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_2 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		batch_x = np.vstack((theta_1, theta_2))
		batch_x = batch_x.T

		self.testSize = batch_x.shape[0]
		batch_y = np.zeros((self.testSize, self.numOutput))

		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])
			
		return batch_x, batch_y

	def inverseDynamics(self, q, qDot, qDDot):

		M11 = self.mLink[0] * (self.link_length[0]**2)/4 + self.inertia[0] + self.mLink[1]*(self.link_length[0]**2 + 0.25*self.link_length[1]**2 + self.link_length[0]*self.link_length[1]*cos(q[1])) + self.inertia[1]
		M22 = 0.25*self.mLink[1]*self.link_length[1]**2 + self.inertia[1]
		M12 = self.mLink[1]*(0.25*self.link_length[1]**2 + 0.5*self.link_length[0]*self.link_length[1]*cos(q[1])) + self.inertia[1]

		h = 0.5 * self.mLink[1] * self.link_length[0] * self.link_length[1] * sin(q[1])

		G1 = self.mLink[0] * self.link_length[0] *self.gravity*cos(q[0]) + self.mLink[1]*self.gravity*(0.5*self.link_length[1]*cos(q[0] + q[1]) + self.link_length[0] * cos(q[0])) 
		G2 = 0.5*self.mLink[1] * self.gravity * self.link_length[1] * cos(q[0] + q[1])

		torques = np.zeros(self.dof)
		torques[0] = M11 * qDDot[0] + M12 * qDDot[1] - h*qDot[1]**2 - 2*h * qDot[0] * qDot[1] + G1
		torques[1] = M22 * qDDot[1] + M12 * qDDot[1] + h*qDot[0]**2 + G2
		
		return torques

	def generateInverseDynamicsData(self, batchSize):
		np.random.seed(1)
		q = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		qDot = 2.0*np.random.rand(batchSize, self.dof) - 1.0*np.ones([batchSize, self.dof])
		qDDot = 2.0*np.random.rand(batchSize, self.dof) - 1.0*np.ones([batchSize, self.dof])
		
		torques = np.zeros((batchSize, self.numOutput))
		
		for i in range(batchSize):
			torques[i, :] = self.inverseDynamics(q[i,:], qDot[i,:], qDDot[i,:])

		return np.hstack((np.hstack((q, qDot)), qDDot)), torques

	def generateInverseDynamicsTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_2 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		q = np.vstack((theta_1, theta_2))
		q = q.T

		self.testSize = q.shape[0]
		qDot = 2.0*np.random.rand(self.testSize, self.dof) - 1.0*np.ones([self.testSize, self.dof])
		qDDot = 2.0*np.random.rand(self.testSize, self.dof) - 1.0*np.ones([self.testSize, self.dof])

		torques = np.zeros((self.testSize, self.numOutput))

		for i in range(self.testSize):
			torques[i, :] = self.inverseDynamics(q[i,:], qDot[i,:], qDDot[i,:])
			
		return np.hstack((np.hstack((q, qDot)), qDDot)), torques


class PlanarRRR(object):
	"""Planar RRR manipulator"""
	def __init__(self, params):
		self.dof = params['dof']
		self.link_length = params['linkLengths']
		self.mLink = params['mLink']
		self.inertia = params['inertia']
		self.gravity = params['gravity']

	def forwardKinematics(self, angles):
		x = self.link_length[0]*cos(angles[0]) + self.link_length[1]*cos(angles[0] + angles[1]) + self.link_length[2]*cos(angles[0] + angles[1] + angles[2])
		y = self.link_length[0]*sin(angles[0]) + self.link_length[1]*sin(angles[0] + angles[1]) + self.link_length[2]*sin(angles[0] + angles[1] + angles[2])
		theta = (angles[0] + angles[1] + angles[2])%(2*pi)

		endEffector = np.array([x, y, theta])
		return endEffector

	def generateFKData(self, batchSize):
		# np.random.seed(1)
		batch_x = pi*np.random.rand(batchSize, self.dof) - 0.*pi*np.ones([batchSize, self.dof])
		# batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros((batchSize, self.numOutput))
		
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateFKTrajectory(self):
		theta_1 = np.hstack((np.linspace(0.1,0.1,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(0.9*pi,0.9*pi,700)))
		theta_2 = np.hstack((np.linspace(0.1,0.1,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(0.9*pi,0.9*pi,100)))
		# theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		theta_3 = np.hstack((np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100)))
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,7,1)))
		batch_x = batch_x.T

		self.testSize = batch_x.shape[0]
		batch_y = np.zeros((self.testSize, self.numOutput))

		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])
			
		return batch_x, batch_y

	def generateIKData(self, batchSize):
		batch_y, batch_x = self.generateFKData(batchSize)
		return batch_x, batch_y

	def generateIKTrajectory(self):
		batch_y, batch_x = self.generateFKTrajectory()
		return batch_x, batch_y

class NonPlanarRRR(object):
	"""Planar RRR manipulator"""
	def __init__(self, params):
		self.dof = params['dof']
		self.link_length = params['linkLengths']
		self.mLink = params['mLink']
		self.inertia = params['inertia']
		self.gravity = params['gravity']

	def forwardKinematics(self, angles):
		# RRR 3D manipulator
		r = self.link_length[1]*cos(angles[1]) + self.link_length[2]*cos(angles[1] + angles[2])
		z = self.link_length[1]*sin(angles[1]) + self.link_length[2]*sin(angles[1] + angles[2])
		
		x = r*cos(angles[0])
		y = r*sin(angles[0])
		endEffector = np.array([x, y, z])
		return endEffector

	def generateFKData(self, batchSize):
		np.random.seed(1)
		batch_x = pi*np.random.rand(batchSize, self.dof)
		batch_y = np.zeros((batchSize, self.numOutput))

		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateFKTrajectory(self):
		theta_1 = np.hstack((np.linspace(0.1,0.1,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(0.9*pi,0.9*pi,700)))
		theta_2 = np.hstack((np.linspace(0.1,0.1,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(0.9*pi,0.9*pi,100)))
		# theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		theta_3 = np.hstack((np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100), np.linspace(0.1,0.9*pi,100)))
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,7,1)))
		batch_x = batch_x.T

		self.testSize = batch_x.shape[0]
		batch_y = np.zeros((self.testSize, self.numOutput))
		
		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def dynamics(self, q, qDot, qDDot):

		M11 = self.mLink[0] * (self.link_length[0]**2)/4 + self.inertia[0] + self.mLink[1]*(self.link_length[0]**2 + 0.25*self.link_length[1]**2 + self.link_length[0]*self.link_length[1]*cos(q[1])) + self.inertia[1]
		M22 = 0.25*self.mLink[1]*self.link_length[1]**2 + self.inertia[1]
		M12 = self.mLink[1]*(0.25*self.link_length[1]**2 + 0.5*self.link_length[0]*self.link_length[1]*cos(q[1])) + self.inertia[1]

		h = 0.5 * self.mLink[1] * self.link_length[0] * self.link_length[1] * sin(q[1])

		G1 = self.mLink[0] * self.link_length[0] *self.gravity*cos(q[0]) + self.mLink[1]*self.gravity*(0.5*self.link_length[1]*cos(q[0] + q[1]) + self.link_length[0] * cos(q[0])) 
		G2 = 0.5*self.mLink[1] * self.gravity * self.link_length[1] * cos(q[0] + q[1])

		torques = np.zeros(self.dof)
		torques[0] = M11 * qDDot[0] + M12 * qDDot[1] - h*qDot[1]**2 - 2*h * qDot[0] * qDot[1] + G1
		torques[1] = M22 * qDDot[1] + M12 * qDDot[1] + h*qDot[0]**2 + G2
		
		return torques

	def generateDynamicsData(self, batchSize):
		np.random.seed(1)
		q = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		qDot = 2.0*np.random.rand(batchSize, self.dof) - 1.0*np.ones([batchSize, self.dof])
		qDDot = 2.0*np.random.rand(batchSize, self.dof) - 1.0*np.ones([batchSize, self.dof])
		
		torques = np.zeros((batchSize, self.numOutput))
		
		for i in range(batchSize):
			torques[i, :] = self.dynamics(q[i,:], qDot[i,:], qDDot[i,:])

		return np.hstack((np.hstack((q, qDot)), qDDot)), torques

	def generateDynamicsTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_2 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		q = np.vstack((theta_1, theta_2))
		q = q.T

		self.testSize = q.shape[0]
		qDot = 2.0*np.random.rand(self.testSize, self.dof) - 1.0*np.ones([self.testSize, self.dof])
		qDDot = 2.0*np.random.rand(self.testSize, self.dof) - 1.0*np.ones([self.testSize, self.dof])

		torques = np.zeros((self.testSize, self.numOutput))

		for i in range(self.testSize):
			torques[i, :] = self.dynamics(q[i,:], qDot[i,:], qDDot[i,:])
			
		return np.hstack((np.hstack((q, qDot)), qDDot)), torques

	def generateIKData(self, batchSize):
		batch_y, batch_x = self.generateFKData(batchSize)
		return batch_x, batch_y

	def generateIKTrajectory(self):
		batch_y, batch_x = self.generateFKTrajectory()
		return batch_x, batch_y

class NonPlanarRRRRR(object):
	"""docstring for ClassName"""
	def __init__(self, params):
		self.dof = params['dof']
		self.mLink = params['mLink']
		self.inertia = params['inertia']
		self.gravity = params['gravity']
		self.link_length = (params['dh_parameters'][:, 0]).reshape([1,self.dof])
		self.link_twist = (params['dh_parameters'][:, 1]).reshape([1,self.dof])
		self.link_offset = (params['dh_parameters'][:, 2]).reshape([1,self.dof])
		self.joint_angle = (params['dh_parameters'][:, 3]).reshape([1,self.dof])


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

	def generateFKData(self, batchSize):
		batch_x = 2.0*pi*np.random.rand(batchSize, self.dof) - pi*np.ones([batchSize, self.dof])
		batch_y = np.zeros((batchSize, self.numOutput))
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateFKTrajectory(self):
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

class NonPlanarRRRR(object):
	"""docstring for ClassName"""
	def __init__(self, params):
		self.dof = params['dof']
		self.mLink = params['mLink']
		self.inertia = params['inertia']
		self.gravity = params['gravity']
		self.link_length = (params['dh_parameters'][:, 0]).reshape([1,self.dof])
		self.link_twist = (params['dh_parameters'][:, 1]).reshape([1,self.dof])
		self.link_offset = (params['dh_parameters'][:, 2]).reshape([1,self.dof])
		self.joint_angle = (params['dh_parameters'][:, 3]).reshape([1,self.dof])

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
		jointAngles = np.append(jointAngles,0)
		for index in range(self.dof):
			transform = self.dhMatrix(index, jointAngles)
			if index != 0:
				frames[:, :, index] = np.matmul(frames[:, :, index - 1], transform)
			else:
				frames[:, :, index] = transform

		endEffector = frames[0:3, 3, self.dof-1]
		return endEffector

	def generateFKData(self, batchSize):
		batch_x = 2.0*pi*np.random.rand(batchSize, self.dof-1) - pi*np.ones([batchSize, self.dof-1])
		batch_y = np.zeros((batchSize, self.numOutput))
		for i in range(batchSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])

		return batch_x, batch_y

	def generateFKTrajectory(self):
		theta_1 = np.hstack((np.linspace(0,0,700), np.linspace(0.5,0.5,700), np.linspace(1.,1.,700), np.linspace(1.5,1.5,700), np.linspace(2.,2.,700), np.linspace(2.5,2.5,700), np.linspace(pi,pi,700)))
		theta_2 = np.hstack((np.linspace(0,0,100), np.linspace(0.5,0.5,100), np.linspace(1.,1.,100), np.linspace(1.5,1.5,100), np.linspace(2.,2.,100), np.linspace(2.5,2.5,100), np.linspace(pi,pi,100)))
		theta_3 = np.hstack((np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100), np.linspace(0,pi,100)))
		theta_4 = np.linspace(0.2,0.2,700)
		# theta_5 = np.linspace(0.2,0.2,700)
		
		batch_x = np.vstack((theta_2, theta_3))
		batch_x = np.vstack((theta_1, np.repeat(batch_x,7,1)))
		batch_x = np.vstack((batch_x, np.repeat(theta_4,7,0)))
		# batch_x = np.vstack((batch_x, np.repeat(theta_5,7,0)))
		batch_x = batch_x.T
		
		self.testSize = batch_x.shape[0]
		batch_y = np.zeros([self.testSize, self.numOutput])
		
		for i in range(self.testSize):
			batch_y[i, :] = self.forwardKinematics(batch_x[i,:])
			
		return batch_x, batch_y