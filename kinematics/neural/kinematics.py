import math
import numpy as np
from collections import namedtuple
from numpy.linalg import inv

class Kinematics:
	def __init__(self):
		self.offset = np.array([[0.,	 0.,	0.,		0.,		0.,     0.],
						   [0,-0.006, 0.031, -0.031, -0.038,    0.],
						   [0,  0.07,  0.28,   0.28,  0.055, 0.032]])

		self.L1 = 0.28
		self.L2 = 0.28

	def inverseKinematics(x, y, z, theta): # end effector is always pointing down
		if z < -0.03:
			raise ValueError('Point is inside of the table')

		angles = np.zeros((5,1))

		combined_x_offset = 0.044

		tool_z_offset = 0.032
		module5_z_offset = 0.055;
		module1_z_offset = 0.;
		module2_z_offset = 0.07;

		z_mod = (-module2_z_offset - module1_z_offset + z + tool_z_offset
			 + module5_z_offset)

		r = np.sqrt(x**2+y**2+z_mod**2)
		if r > 0.572*0.95:
			raise ValueError('Point is outside of the reachable workspace')

		r1 = np.sqrt(x**2+y**2)
		if r1 > combined_x_offset:
			d = np.sqrt(r1**2 - combined_x_offset**2)
		else:
			raise ValueError('Radius of arm is too close to the base')

		angles[0] = (np.pi/2 + np.arccos((d**2 - combined_x_offset**2 + r1**2)
					/ (2*d*r1))- np.arcsin(x/r1))

		r2 = np.sqrt(z_mod**2 + d**2)

		angles[1] = -(-np.arcsin(z_mod/r2) + np.pi/2
					 - np.arccos((self.L1**2-self.L2**2+r2**2)/(2*self.L1*r2)))
		angles[2] = np.pi-np.arccos((self.L1**2+self.L2**2-r2**2)/(2*self.L1*self.L2))

		angles[3] = -(np.pi + angles[1] - angles[2])
		angles[4] = -(np.pi/2 - angles[0]) + theta

		return angles

	def forwardKinematics(angles):
		H_ground_to_joint1 = Tz(angles[0],self.offset[:,0])
		H_ground_to_joint2 = np.dot(H_ground_to_joint1,Ty(-angles[1],self.offset[:,1]))
		H_ground_to_joint3 = np.dot(H_ground_to_joint2,Ty(angles[2],self.offset[:,2]))
		H_ground_to_joint4 = np.dot(H_ground_to_joint3,Ty(-angles[3],self.offset[:,3]))
		H_ground_to_joint5 = np.dot(H_ground_to_joint4,Tz(angles[4],self.offset[:,4]))
		return np.around(np.dot(H_ground_to_joint2,T(self.offset[:,5])),decimals=5)

	def Ty(theta, trans):
		return np.array([[np.cos(theta),  0., np.sin(theta), 	trans[0]],
						 [		0.,  	  1., 		0.,			trans[1]],
						 [-np.sin(theta), 0., np.cos(theta), 	trans[2]],
						 [		0., 	  0., 		0., 		 	  1]])
	def Tz(theta,trans):
		return np.array([[np.cos(theta), -np.sin(theta), 0., trans[0]],
						 [np.sin(theta),  np.cos(theta), 0., trans[1]],
						 [ 		0., 				0.,  1., trans[2]],
						 [		0., 				0.,  0., 	   1.]])
	def T(trans):
		return np.array([[1., 0.,  0., trans[0]],
						 [0., 1.,  0., trans[1]],
						 [0., 0.,  1., trans[2]],
						 [0., 0.,  0., 	   1.]])
