import math
import numpy as np
from collections import namedtuple
from numpy.linalg import inv

from kinematics import Kinematics as KMTCS

def generateTrajectory(posList,dt):
	trajectory = namedtuple('Trajectory',['pos','vel','acc','time'])
	trajectory.time = np.arange(0,posList.x.size*dt,dt)
	
	trajectory.pos = np.vstack((posList.x,posList.y,posList.z))

	trajectory.vel = np.vstack((differentiate(posList.x,dt),
								differentiate(posList.y,dt),
								differentiate(posList.z,dt)))

	trajectory.acc = np.vstack((differentiate(trajectory.vel_x,dt),
								differentiate(trajectory.vel_y,dt),
								differentiate(trajectory.vel_z,dt)))

	return trajectory

def generateJointTrajectory(posList,dt):
	jointTrajectory = namedtuple('JointTrajectory',['ang','vel','acc','time'])
	theta = np.zeros(posList.x.shape)

	jointTrajectory.ang = KMTCS.inverseKinematics(posList.x[0],posList.y[0],posList.z[0],theta[0])

	for i in range(1,posList.x.size):
		angles = KMTCS.inverseKinematics(posList.x[i],posList.y[i],posList.z[i],theta[i])
		jointTrajectory.ang = np.append(jointTrajectory.ang,angles,axis=1)

	jointTrajectory.vel = np.zeros(jointTrajectory.ang.shape)
	jointTrajectory.acc = np.zeros(jointTrajectory.ang.shape)

	for i in range(5):
		jointTrajectory.vel[i] = differentiate(jointTrajectory.ang[i],dt)
		jointTrajectory.acc[i] = differentiate(jointTrajectory.vel[i],dt)

	return jointTrajectory

def generateJointTrajectory_now(time,traj_type='circle',tf=2.5):
	theta = np.zeros((1,1))
	dt = 0.0001
	angles = np.zeros((5,5))
	vel = np.zeros((3,5))

	if traj_type == 'circle':
		c_x = 0.0
		c_y = 0.3
		c_z = 0.
		radius = 0.095

		for i in range(5):
			peturb = dt*float(i-2)
			x,y,z = circle_now(c_x,c_y,c_z,radius,time+peturb,tf)
			ang_temp = KMTCS.inverseKinematics(x,y,z,theta)
			for j in range(5):
				angles[i,j] = ang_temp[j]

		for i in range(3):
			vel[i] = (angles[:][i+2]-angles[:][i])/(2*dt)

		acc = (vel[2]-vel[0])/(2*dt)

	else:
		msg = 'Cannot generate trajectory for unknown type: ' + traj_type
		raise ValueError(msg)

	return angles[2],vel[1],acc

def differentiate(vector,dt):
	diff_vec = np.zeros(vector.shape)
	diff_vec[0] = (vector[0]-vector[1])/dt
	diff_vec[-1] = (vector[-2]-vector[-1])/dt
	for i in range(1,vector.size-1):
		diff_vec[i] = (vector[i+1]-vector[i-1])/(2*dt)

	return diff_vec

def circle(x,y,z,radius,tf,dt=0.01):
	time = np.arange(0,tf,dt)
	points = namedtuple('PointList',['x','y','z'])
	points.x = x + radius*np.cos(2*np.pi*time/tf)
	points.y = y + radius*np.sin(2*np.pi*time/tf)
	points.z = z + np.zeros(time.shape)
	return points

def circle_now(x,y,z,radius,time,tf):
	x = x + radius*np.sin(2*np.pi*time/tf)
	y = y + radius*np.cos(2*np.pi*time/tf)
	z = z
	return x,y,z

def minJerkSetup(x0,t0,tf):
	A = np.array([[1,t0, t0**2, t0**3, t0**4, t0**5],
					[0,1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
					[0,0, 2, 6*t0, 12*t0**2, 20*t0**3],
					[1,tf, tf**2, tf**3, tf**4, tf**5],
					[0,1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
					[0,0, 2, 6*tf, 12*tf**2, 20*tf**3]])
	constants = np.dot(inv(A),x0)
	return constants

def minJerkStep(t,constants):
	pos = np.dot(np.array([1,t, t**2, t**3, t**4, t**5]),constants)
	vel = np.dot(np.array([0,1, 2*t, 3*t**2, 4*t**3, 5*t**4]),constants)
	accel = np.dot(np.array([0,0, 2, 6*t, 12*t**2, 20*t**3]),constants)
	return pos,vel,accel

def minJerkSetup_now(initial_angles,tf,waypoints,t_array=None):
	num_waypoints = waypoints.shape[1]
	try:
		if t_array == None:
			del_t = float(tf)/float(num_waypoints)
			t_array = del_t*np.ones((num_waypoints,1))
		elif not t_array.size == num_waypoints:
			raise ValueError('Time array length is incorrect')
		elif not tf == np.sum(t_array):
			raise ValueError('Time array must add up to final time')
	except:
		if not t_array.size == num_waypoints:
			raise ValueError('Time array length is incorrect')
		elif not tf == np.sum(t_array):
			raise ValueError('Time array must add up to final time')

	joint_constants = namedtuple('joint_constants','J1 J2 J3 J4 J5')
	joint_const = joint_constants(np.zeros((6,num_waypoints)),
								np.zeros((6,num_waypoints)),
								np.zeros((6,num_waypoints)),
								np.zeros((6,num_waypoints)),
								np.zeros((6,num_waypoints)))
	
	x0 = np.zeros((5,6))
	if initial_angles.ndim == 2:
		if initial_angles.shape[0] == 5:
			initial_angles = initial_angles.T
	x0[:,0] = initial_angles
	x0[:,3] = KMTCS.inverseKinematics(waypoints[0,0],
								   waypoints[1,0],
								   waypoints[2,0],
								   waypoints[3,0]).T

	t0 = np.zeros((num_waypoints,1))
	tf = np.zeros((num_waypoints,1))
	tf[0] = t_array[0]

	for i in range(num_waypoints):
		if i > 0:
			x0[:,0] = x0[:,3]
			x0[:,3] = KMTCS.inverseKinematics(waypoints[0,i],
										   waypoints[1,i],
										   waypoints[2,i],
										   waypoints[3,i]).T
			t0[i] = tf[i-1]
			tf[i] = t0[i]+t_array[i]
		joint_const.J1[:,i] = minJerkSetup(x0[0],t0[i],tf[i])
		joint_const.J2[:,i] = minJerkSetup(x0[1],t0[i],tf[i])
		joint_const.J3[:,i] = minJerkSetup(x0[2],t0[i],tf[i])
		joint_const.J4[:,i] = minJerkSetup(x0[3],t0[i],tf[i])
		joint_const.J5[:,i] = minJerkSetup(x0[4],t0[i],tf[i])
	
	return joint_const

def minJerkStep_now(time,tf,waypoints,joint_const,t_array=None):
	num_waypoints = waypoints.shape[1]
	try:
		if t_array == None:
			del_t = float(tf)/float(num_waypoints)
			t_array = del_t*np.ones((num_waypoints,1))
			k = int(np.floor(time/(del_t)))
		else:
			sum_time = 0.
			k = 0
			for i in range(t_array.size-1):
				sum_time = sum_time + t_array[i]
				if time < sum_time:
					break
				k = k+1
	except:
		sum_time = 0.
		k = 0
		for i in range(t_array.size-1):
			sum_time = sum_time + t_array[i]
			if time < sum_time:
				break
			k = k+1

	if not t_array.size == num_waypoints:
		raise ValueError('Time array is length is incorrect')
	if not tf == np.sum(t_array):
		raise ValueError('Time array must add up to final time')
	pos = np.zeros((5,1))
	vel = np.zeros((5,1))
	accel = np.zeros((5,1))

	for j in range(5):
		pos[j],vel[j],accel[j] = minJerkStep(time,joint_const[j][:,k])

	return pos,vel,accel