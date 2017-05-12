#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import std_msgs.msg
import arm_planner.srv
import hebi_cpp.msg
import hebi_cpp.srv
import numpy as np
import matplotlib.pyplot as plt
import GPy
import csv
import math
from datetime import datetime
import time
from numpy.linalg import inv
import rosbag
import os
import TrajectoryGenerartor as TG

# global ML_count
# global RMSE_ID
# global RMSE_ML1
# global save_position

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

def superpositionSine(t,amp=[np.pi/16,np.pi/12],f=[0.6,0.4],phi=[0,0]):

	pos = (amp[0]*np.sin(2.*np.pi*f[0]*t+phi[0])-
			amp[0]*np.sin(phi[0])+
			amp[1]*np.sin(2.*np.pi*f[1]*t+phi[1])-
			amp[1]*np.sin(phi[1]))
	vel = (2.*np.pi*(amp[0]*f[0]*np.cos(2.*np.pi*f[0]*t+phi[0])
			+amp[1]*f[1]*np.cos(2*np.pi*f[1]*t+phi[1])))
	accel = (-4.*np.pi**2*(amp[0]*f[0]**2*np.sin(2.*np.pi*f[0]*t+phi[0])
			+amp[1]*f[1]**2*np.sin(2.*np.pi*f[1]*t+phi[1])))

	return pos,vel,accel

def checkMinMax(Min,Max,Test):
	if Min > Test:
		Min = Test
	if Max < Test:
		Max = Test
	if Min > Max:
		raise ValueError("Min cannot be greater than Max")

	return Min, Max

def UpdateDataSet(data,fbk,New=True,min_limit=0):
	Min = fbk.time.size
	Max = fbk.time.size
	Min,Max = checkMinMax(Min,Max,fbk.position.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.velocity.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.accel.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.torque.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.positionCmd.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.velocityCmd.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.accelCmd.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.deflection.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.deflection_vel.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.velocityFlt.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.motorSensorTemperature.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.windingTemp.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.windingTempFlt.size/5)
	Min,Max = checkMinMax(Min,Max,fbk.torqueCmd.size/5)

	Min,Max = checkMinMax(Min,Max,fbk.torqueID.size/5)
	if Max-Min>10:
		print Max-Min, "samples had to be dropped to equalize set"
	max_limit = Min
	if New:
		if max_limit>0:
			data.time = np.array(fbk.time[min_limit:max_limit])

			data.position = np.asarray((fbk.position[:,min_limit:max_limit]))
			data.velocity = np.asarray((fbk.velocity[:,min_limit:max_limit]))
			data.torque = np.asarray((fbk.torque[:,min_limit:max_limit]))

			data.positionCmd = np.asarray((fbk.positionCmd[:,min_limit:max_limit]))
			data.velocityCmd = np.asarray((fbk.velocityCmd[:,min_limit:max_limit]))
			data.accelCmd = np.asarray((fbk.accelCmd[:,min_limit:max_limit]))

			data.deflection = np.asarray((fbk.deflection[:,min_limit:max_limit]))
			data.deflection_vel = np.asarray((fbk.deflection_vel[:,min_limit:max_limit]))
			data.velocityFlt = np.asarray((fbk.velocityFlt[:,min_limit:max_limit]))
			data.motorSensorTemperature = np.asarray((fbk.motorSensorTemperature[:,min_limit:max_limit]))
			data.windingTemp = np.asarray((fbk.windingTemp[:,min_limit:max_limit]))
			data.windingTempFlt = np.asarray((fbk.windingTempFlt[:,min_limit:max_limit]))
			data.torqueCmd = np.asarray((fbk.torqueCmd[:,min_limit:max_limit]))
			data.torqueID = np.asarray((fbk.torqueID[:,min_limit:max_limit]))
			data.epsTau = np.asarray((fbk.epsTau[:,min_limit:max_limit]))
			data.accel = np.asarray((fbk.accel[:,min_limit:max_limit]))

		else:
			data.time = np.array(fbk.time)
			data.position = np.asarray((fbk.position))
			data.velocity = np.asarray((fbk.velocity))
			data.torque = np.asarray((fbk.torque))

			data.positionCmd = np.asarray((fbk.positionCmd))
			data.velocityCmd = np.asarray((fbk.velocityCmd))
			data.accelCmd = np.asarray((fbk.accelCmd))

			data.deflection = np.asarray((fbk.deflection))
			data.deflection_vel = np.asarray((fbk.deflection_vel))
			data.velocityFlt = np.asarray((fbk.velocityFlt))
			data.motorSensorTemperature = np.asarray((fbk.motorSensorTemperature))
			data.windingTemp = np.asarray((fbk.windingTemp))
			data.windingTempFlt = np.asarray((fbk.windingTempFlt))
			data.torqueCmd = np.asarray((fbk.torqueCmd))
			data.torqueID = np.asarray((fbk.torqueID))
			data.epsTau = np.asarray((fbk.epsTau))
			data.accel = np.asarray((fbk.accel))
						
	else:
		if max_limit>min_limit:
			data.time = np.hstack((data.time,fbk.time[min_limit:max_limit]))
			data.position = np.hstack((data.position,np.asarray((fbk.position[:,min_limit:max_limit]))))
			data.velocity = np.hstack((data.velocity,np.asarray((fbk.velocity[:,min_limit:max_limit]))))
			data.torque = np.hstack((data.torque,np.asarray((fbk.torque[:,min_limit:max_limit]))))

			data.positionCmd = np.hstack((data.positionCmd,np.asarray((fbk.positionCmd[:,min_limit:max_limit]))))
			data.velocityCmd = np.hstack((data.velocityCmd,np.asarray((fbk.velocityCmd[:,min_limit:max_limit]))))
			data.accelCmd = np.hstack((data.accelCmd,np.asarray((fbk.accelCmd[:,min_limit:max_limit]))))

			data.deflection = np.hstack((data.deflection,np.asarray((fbk.deflection[:,min_limit:max_limit]))))
			data.deflection_vel = np.hstack((data.deflection_vel,np.asarray((fbk.deflection_vel[:,min_limit:max_limit]))))
			data.velocityFlt = np.hstack((data.velocityFlt,np.asarray((fbk.velocityFlt[:,min_limit:max_limit]))))
			data.motorSensorTemperature = np.hstack((data.motorSensorTemperature,np.asarray((fbk.motorSensorTemperature[:,min_limit:max_limit]))))
			data.windingTemp = np.hstack((data.windingTemp,np.asarray((fbk.windingTemp[:,min_limit:max_limit]))))
			data.windingTempFlt = np.hstack((data.windingTempFlt,np.asarray((fbk.windingTempFlt[:,min_limit:max_limit]))))
			data.torqueCmd = np.hstack((data.torqueCmd,np.asarray((fbk.torqueCmd[:,min_limit:max_limit]))))
			data.torqueID = np.hstack((data.torqueID,np.asarray((fbk.torqueID[:,min_limit:max_limit]))))
			data.epsTau = np.hstack((data.epsTau,np.asarray((fbk.epsTau[:,min_limit:max_limit]))))
			data.accel = np.hstack((data.accel,np.asarray((fbk.accel[:,min_limit:max_limit]))))
		else:
			data.time = np.hstack((data.time,fbk.time))
			data.position = np.hstack((data.position,np.asarray((fbk.position))))
			data.velocity = np.hstack((data.velocity,np.asarray((fbk.velocity))))
			data.torque = np.hstack((data.torque,np.asarray((fbk.torque))))

			data.positionCmd = np.hstack((data.positionCmd,np.asarray((fbk.positionCmd))))
			data.velocityCmd = np.hstack((data.velocityCmd,np.asarray((fbk.velocityCmd))))
			data.accelCmd = np.hstack((data.accelCmd,np.asarray((fbk.trajectoryCmd.accelCmd))))

			data.deflection = np.hstack((data.deflection,np.asarray((fbk.deflection))))
			data.deflection_vel = np.hstack((data.deflection_vel,np.asarray((fbk.deflection_vel))))
			data.velocityFlt = np.hstack((data.velocityFlt,np.asarray((fbk.velocityFlt))))
			data.motorSensorTemperature = np.hstack((data.motorSensorTemperature,np.asarray((fbk.motorSensorTemperature))))
			data.windingTemp = np.hstack((data.windingTemp,np.asarray((fbk.windingTemp))))
			data.windingTempFlt = np.hstack((data.windingTempFlt,np.asarray((fbk.windingTempFlt))))
			data.torqueCmd = np.hstack((data.torqueCmd,np.asarray((fbk.torqueCmd))))
			data.torqueID = np.hstack((data.torqueID,np.asarray((fbk.torqueID))))
			data.epsTau = np.hstack((data.epsTau,np.asarray((fbk.epsTau))))
			data.accel = np.hstack((data.accel,np.asarray((fbk.accel))))

	return data

def AverageDataSet(fbk,index_set):
	data = dataStruct()
	data.time = np.mean(fbk.time[index_set[0]:index_set[1]])
	data.position = np.mean(fbk.position[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.velocity = np.mean(fbk.velocity[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.torque = np.mean(fbk.torque[:,index_set[0]:index_set[1]],axis=1,keepdims=True)

	data.positionCmd = np.mean(fbk.positionCmd[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.velocityCmd = np.mean(fbk.velocityCmd[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.accelCmd = np.mean(fbk.accelCmd[:,index_set[0]:index_set[1]],axis=1,keepdims=True)

	data.deflection = np.mean(fbk.deflection[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.deflection_vel = np.mean(fbk.deflection_vel[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.velocityFlt = np.mean(fbk.velocityFlt[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.motorSensorTemperature = np.mean(fbk.motorSensorTemperature[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.windingTemp = np.mean(fbk.windingTemp[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.windingTempFlt = np.mean(fbk.windingTempFlt[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.torqueCmd = np.mean(fbk.torqueCmd[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.torqueID = np.mean(fbk.torqueID[:,index_set[0]:index_set[1]],axis=1,keepdims=True)
	data.accel = np.mean(fbk.accel[:,index_set[0]:index_set[1]],axis=1,keepdims=True)

	length = len(index_set)
	for i in range(2,length):
		data.time = np.hstack((data.time,np.mean(fbk.time[index_set[i-1]:index_set[i]])))
		data.position = np.hstack((data.position,np.mean(fbk.position[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.velocity = np.hstack((data.velocity,np.mean(fbk.velocity[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.torque = np.hstack((data.torque,np.mean(fbk.torque[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))

		data.positionCmd = np.hstack((data.positionCmd,np.mean(fbk.positionCmd[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.velocityCmd = np.hstack((data.velocityCmd,np.mean(fbk.velocityCmd[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.accelCmd = np.hstack((data.accelCmd,np.mean(fbk.accelCmd[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))

		data.deflection = np.hstack((data.deflection,np.mean(fbk.deflection[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.deflection_vel = np.hstack((data.deflection_vel,np.mean(fbk.deflection_vel[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.velocityFlt = np.hstack((data.velocityFlt,np.mean(fbk.velocityFlt[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		
		data.motorSensorTemperature = np.hstack((data.motorSensorTemperature,np.mean(fbk.motorSensorTemperature[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.windingTemp = np.hstack((data.windingTemp,np.mean(fbk.windingTemp[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.windingTempFlt = np.hstack((data.windingTempFlt,np.mean(fbk.windingTempFlt[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.torqueCmd = np.hstack((data.torqueCmd,np.mean(fbk.torqueCmd[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.torqueID = np.hstack((data.torqueID,np.mean(fbk.torqueID[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))
		data.accel = np.hstack((data.accel,np.mean(fbk.accel[:,index_set[i-1]:index_set[i]],axis=1,keepdims=True)))

	if index_set[length-1] == fbk.time.size:
		data.time = np.hstack((data.time,np.mean(fbk.time[index_set[length-1]:-1])))
		data.position = np.hstack((data.position,np.mean(fbk.position[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.velocity = np.hstack((data.velocity,np.mean(fbk.velocity[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.torque = np.hstack((data.torque,np.mean(fbk.torque[:,index_set[length-1]:-1],axis=1,keepdims=True)))

		data.positionCmd = np.hstack((data.positionCmd,np.mean(fbk.positionCmd[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.velocityCmd = np.hstack((data.velocityCmd,np.mean(fbk.velocityCmd[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.accelCmd = np.hstack((data.accelCmd,np.mean(fbk.accelCmd[:,index_set[length-1]:-1],axis=1,keepdims=True)))

		data.deflection = np.hstack((data.deflection,np.mean(fbk.deflection[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.deflection_vel = np.hstack((data.deflection_vel,np.mean(fbk.deflection_vel[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.velocityFlt = np.hstack((data.velocityFlt,np.mean(fbk.velocityFlt[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		
		data.motorSensorTemperature = np.hstack((data.motorSensorTemperature,np.mean(fbk.motorSensorTemperature[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.windingTemp = np.hstack((data.windingTemp,np.mean(fbk.windingTemp[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.windingTempFlt = np.hstack((data.windingTempFlt,np.mean(fbk.windingTempFlt[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.torqueCmd = np.hstack((data.torqueCmd,np.mean(fbk.torqueCmd[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.torqueID = np.hstack((data.torqueID,np.mean(fbk.torqueID[:,index_set[length-1]:-1],axis=1,keepdims=True)))
		data.accel = np.hstack((data.accel,np.mean(fbk.accel[:,index_set[length-1]:-1],axis=1,keepdims=True)))


	return data

def resetPosition(motorOn,ps):
	tf = 4.
	start_time = rospy.Time.now()
	time_from_start = 0.

	cmd = hebi_cpp.msg.CommandML()
	point = JointTrajectoryPoint()

	c_x = 0.0
	c_y = 0.3
	c_z = 0.1
	radius = 0.095
	point.positions = TG.inverseKinematics(c_x,c_y+radius,c_z,0)
	#point.positions = [1.68437,-0.445017,1.18675,4.7561,1.62]#7.94885]
	point.velocities = [0.,0.,0.,0.,0.]
	cmd.epsTau = [0.,0.,0.,0.,0.]
	cmd.jointTrajectory.points.append(point)
	cmd.motorOn = float(motorOn)
	cmd.controlType = 0.

	while tf- time_from_start> 0:
		current_time = rospy.Time.now()
		cmd.jointTrajectory.header.stamp = current_time
		time_from_start = ((current_time-start_time).secs
						  +(current_time-start_time).nsecs/1000000000.)
		ps.traj_pub.publish(cmd)

class dataStruct:
	def __init__(self):
		self.time = np.empty(shape=(1,))
		self.position = np.empty(shape=(5,1))
		self.positionCmd = np.empty(shape=(5,1))
		self.velocity = np.empty(shape=(5,1))
		self.velocityCmd = np.empty(shape=(5,1))
		self.velocityFlt = np.empty(shape=(5,1))
		self.accel = np.empty(shape=(5,1))
		self.accelCmd = np.empty(shape=(5,1))
		self.torque = np.empty(shape=(5,1))
		self.torqueCmd = np.empty(shape=(5,1))
		self.torqueID = np.empty(shape=(5,1))
		self.deflection = np.empty(shape=(5,1))
		self.deflection_vel = np.empty(shape=(5,1))
		self.motorSensorTemperature = np.empty(shape=(5,1))
		self.windingTemp = np.empty(shape=(5,1))
		self.windingTempFlt = np.empty(shape=(5,1))
		self.epsTau = np.empty(shape=(5,1))

class pubSub:
	def __init__(self,bag=False,folderName=None,flush=False,control_info={}):
		self.queue = dataStruct()
		self.restart_joint = True
		self.restart_traj = True
		self.restart_arm = True
		self.count = 1
		self.minimum = 0
		self.initial_time = 0.
		self.prev_time = 0.
		self.final_time = 999.
		self.startBag = False
		self.bagging = False
		self.flush = flush

		self.traj_pub = rospy.Publisher("ml_publisher",
										hebi_cpp.msg.CommandML,queue_size=1)
		self.fbk_sub = rospy.Subscriber("armcontroller_fbk",
							hebi_cpp.msg.FeedbackML,self.armControlCallback,
							queue_size=5000)
		self.path1_pub = rospy.Publisher("/path1/color",
							std_msgs.msg.ColorRGBA,queue_size=1)
		self.path2_pub = rospy.Publisher("/path2/color",
							std_msgs.msg.ColorRGBA,queue_size=1)

		if bag:
			self.startBag = True
			if not folderName == None:
				if os.path.exists(folderName):
					self.bag_name = (folderName + '/'
									+ control_info['Set'] + '.bag')
				else:
					raise ValueError('The folder does not exist')
			else:
				raise ValueError('If bagging data, a folderName must be given')
			self.bag = rosbag.Bag(self.bag_name, 'w')
		# self.state_sub = rospy.Subscriber("jointState_fbk",
		# 								JointState,self.jointStateCallback,
		# 								queue_size=1000)
		# self.cmd_sub = rospy.Subscriber("trajectoryCmd_fbk",
		# 								JointTrajectoryPoint,
		# 								self.trajectoryCmdCallback,
										# queue_size=1000)


	# def jointStateCallback(self,ros_data):
	# 	if self.restart_joint or self.restart_traj or self.restart_arm:
	# 		self.addToQueue(ros_data,New=True,fbk_type="jointState")
	# 		self.restart_joint = False
	# 	else:
	# 		self.addToQueue(ros_data,New=False,fbk_type="jointState")
	# 		self.count[0] += 1
	# 	self.minimum = min(self.count)-1
		
	# def trajectoryCmdCallback(self,ros_data):
	# 	if self.restart_joint or self.restart_traj or self.restart_arm:
	# 		self.addToQueue(ros_data,New=True,fbk_type="trajectoryCmd")
	# 		self.restart_traj = False
	# 	else:
	# 		self.addToQueue(ros_data,New=False,fbk_type="trajectoryCmd")
	# 		self.count[1] += 1
	# 		print self.count
	# 	self.minimum = min(self.count)-1

	def armControlCallback(self,ros_data):
		if self.restart_arm:
			self.addToQueue(ros_data,New=True,fbk_type="armControl")
			self.restart_arm = False
		elif not self.flush:
			self.addToQueue(ros_data,New=False,fbk_type="armControl")
			self.count += 1
		self.minimum = self.count
		if self.bagging:
			self.bag.write(control_info['Set'],ros_data)
	def addToQueue(self,fbk,New=True,fbk_type="Undefined"):
		if fbk_type.lower() == "jointstate":
			if New:
				time_secs = float(fbk.header.stamp.secs)+float(fbk.header.stamp.nsecs)/1000000000.
				
				self.initial_time = time_secs
				self.queue.time = np.array(time_secs-time_secs)
				self.queue.position = np.array(np.matrix((fbk.position)).T)
				self.queue.velocity = np.array(np.matrix((fbk.velocity)).T)
				self.queue.torque = np.array(np.matrix((fbk.effort)).T)
				
			else:
				if time_secs > self.final_time:
					return
				time_secs = float(fbk.header.stamp.secs)+float(fbk.header.stamp.nsecs)/1000000000. - self.initial_time
				self.queue.time = np.hstack((self.queue.time,np.array(time_secs)))
				self.queue.position = np.hstack((self.queue.position,np.array(np.matrix((fbk.position)).T)))
				self.queue.velocity = np.hstack((self.queue.velocity,np.array(np.matrix((fbk.velocity)).T)))
				self.queue.torque = np.hstack((self.queue.torque,np.array(np.matrix((fbk.effort)).T)))
		elif fbk_type.lower() == "trajectorycmd":
			if New:
				self.queue.positionCmd = np.array(np.matrix((fbk.positions)).T)
				self.queue.velocityCmd = np.array(np.matrix((fbk.velocities)).T)
				self.queue.accelCmd = np.array(np.matrix((fbk.accelerations)).T)
			else:
				if time_secs > self.final_time:
					return
				self.queue.positionCmd = np.hstack((self.queue.positionCmd,np.array(np.matrix((fbk.positions)).T)))
				self.queue.velocityCmd = np.hstack((self.queue.velocityCmd,np.array(np.matrix((fbk.velocities)).T)))
				self.queue.accelCmd = np.hstack((self.queue.accelCmd,np.array(np.matrix((fbk.accelerations)).T)))
		elif fbk_type.lower() == "armcontrol":
			if New:
				time_secs = float(fbk.header.stamp.secs)+float(fbk.header.stamp.nsecs)/1000000000.
				
				self.initial_time = time_secs
				self.queue.time = np.array(time_secs-time_secs)
				self.queue.position = np.array(np.matrix((fbk.jointState.position)).T)
				self.queue.velocity = np.array(np.matrix((fbk.jointState.velocity)).T)
				self.queue.torque = np.array(np.matrix((fbk.jointState.effort)).T)

				self.queue.positionCmd = np.array(np.matrix((fbk.trajectoryCmd.positions)).T)
				self.queue.velocityCmd = np.array(np.matrix((fbk.trajectoryCmd.velocities)).T)
				self.queue.accelCmd = np.array(np.matrix((fbk.trajectoryCmd.accelerations)).T)

				self.queue.deflection = np.array(np.matrix((fbk.deflections)).T)
				self.queue.deflection_vel = np.array(np.matrix((fbk.deflection_vel)).T)
				self.queue.velocityFlt = np.array(np.matrix((fbk.velocityFlt)).T)
				self.queue.motorSensorTemperature = np.array(np.matrix((fbk.motorSensorTemperature)).T)
				self.queue.windingTemp = np.array(np.matrix((fbk.windingTemp)).T)
				self.queue.windingTempFlt = np.array(np.matrix((fbk.windingTempFlt)).T)
				self.queue.torqueCmd = np.array(np.matrix((fbk.torqueCmd)).T)
				self.queue.torqueID = np.array(np.matrix((fbk.torqueID)).T)
				self.queue.epsTau = np.array(np.matrix((fbk.epsTau)).T)
				self.queue.accel = np.array(np.matrix((fbk.accel)).T)
			else:
				time_secs = float(fbk.header.stamp.secs)+float(fbk.header.stamp.nsecs)/1000000000. - self.initial_time
				if time_secs > self.final_time:
					return
				self.queue.time = np.hstack((self.queue.time,np.array(time_secs)))
				self.queue.position = np.hstack((self.queue.position,np.array(np.matrix((fbk.jointState.position)).T)))
				self.queue.velocity = np.hstack((self.queue.velocity,np.array(np.matrix((fbk.jointState.velocity)).T)))
				self.queue.torque = np.hstack((self.queue.torque,np.array(np.matrix((fbk.jointState.effort)).T)))

				self.queue.positionCmd = np.hstack((self.queue.positionCmd,np.array(np.matrix((fbk.trajectoryCmd.positions)).T)))
				self.queue.velocityCmd = np.hstack((self.queue.velocityCmd,np.array(np.matrix((fbk.trajectoryCmd.velocities)).T)))
				self.queue.accelCmd = np.hstack((self.queue.accelCmd,np.array(np.matrix((fbk.trajectoryCmd.accelerations)).T)))

				self.queue.deflection = np.hstack((self.queue.deflection,np.array(np.matrix((fbk.deflections)).T)))
				self.queue.deflection_vel = np.hstack((self.queue.deflection_vel,np.array(np.matrix((fbk.deflection_vel)).T)))
				self.queue.velocityFlt = np.hstack((self.queue.velocityFlt,np.array(np.matrix((fbk.velocityFlt)).T)))
				self.queue.motorSensorTemperature = np.hstack((self.queue.motorSensorTemperature,np.array(np.matrix((fbk.motorSensorTemperature)).T)))
				self.queue.windingTemp = np.hstack((self.queue.windingTemp,np.array(np.matrix((fbk.windingTemp)).T)))
				self.queue.windingTempFlt = np.hstack((self.queue.windingTempFlt,np.array(np.matrix((fbk.windingTempFlt)).T)))
				self.queue.torqueCmd = np.hstack((self.queue.torqueCmd,np.array(np.matrix((fbk.torqueCmd)).T)))
				self.queue.torqueID = np.hstack((self.queue.torqueID,np.array(np.matrix((fbk.torqueID)).T)))
				self.queue.epsTau = np.hstack((self.queue.epsTau,np.array(np.matrix((fbk.epsTau)).T)))
				self.queue.accel = np.hstack((self.queue.accel,np.array(np.matrix((fbk.accel)).T)))
		else:
			raise ValueError("Feedback type for addToQueue is not recognized")
	def unregister(self):
		# self.state_sub.unregister()
		# self.cmd_sub.unregister()
		self.fbk_sub.unregister()
		self.fbk_sub = None

	def reregister(self):
		# self.state_sub = rospy.Subscriber("jointState_fbk",
		# 					JointState,self.jointStateCallback,
		# 					queue_size=1000)
		# self.cmd_sub = rospy.Subscriber("trajectoryCmd_fbk",
		# 					JointTrajectoryPoint,self.trajectoryCmdCallback,
		# 					queue_size=1000)
		self.fbk_sub = rospy.Subscriber("armcontroller_fbk",
							hebi_cpp.msg.FeedbackML,self.armControlCallback,
							queue_size=1000)
		self.restart_joint = True
		self.restart_traj = True
		self.restart_arm = True
		self.queue = dataStruct()
		self.count = 1
		self.initial_time = 0.

	def reset(self):
		self.restart_joint = True
		self.restart_traj = True
		self.restart_arm = True
		self.queue = dataStruct()
		self.count = 1
		self.initial_time = 0.

class modelDatabase:
	def __init__(self,ps,deflection=False,temperature=False):
		self.ps = ps
		self.learn_def = deflection
		self.learn_temp = temperature

		self.train_set = dataStruct()
		self.verify_set = dataStruct()
		self.test_set = dataStruct()
		self.train_mod_set = None
		self.verify_mod_set = None
		self.test_mod_set = None
		self.trained = False
		self.verified = False
		self.tested = False
		self.train_set_size = 0
		self.verify_set_size = 0
		self.downsample_f = 1000.
		self.data_cap = 2000
		self.start_time = 0.  #Changed Start time back to 0 from 0.75
		self.init_position = np.array([0,0,0,0,0])
		self.joints_ML = np.array([0,1,2,3,4])

	def updateSet(self,Set,New=True):
		if Set.lower() == "train":
			if New:
				if self.trained == True:
					print "Overwriting self.trained dataset"
				else:
					self.trained = True
				min_limit = 0
				for i in range(0,self.ps.queue.time.size):
					if self.ps.queue.time[i] > self.start_time:
						min_limit = i
						break
				self.init_position = self.ps.queue.position[:,0]
				self.train_set = UpdateDataSet(self.train_set,self.ps.queue,
								 			   New,min_limit=min_limit)
				if not self.train_mod_set == None:
					self.train_mod_set = None
				self.train_set_size = self.ps.minimum
				# print self.ps.count
				# print self.ps.minimum
			else:
				if self.trained == False:
					raise ValueError('ModelDatabase training set cannot be\
									  updated from an empty set')
				else:

					for i in range(0,self.ps.queue.time.size):
						if self.ps.queue.time[i] > self.start_time:
							min_limit = i
							break
					self.ps.queue.time = (self.ps.queue.time 
										 + self.train_set.time[-1])
					self.train_set = UpdateDataSet(self.train_set,
										self.ps.queue,New,min_limit=min_limit)
					if not self.train_mod_set == None:
						self.train_mod_set = None
					self.train_set_size = self.ps.minimum
		elif Set.lower() == "verify":
			if New:
				if self.verified == True:
					print "Overwriting verification dataset"
				else:
					self.verified = True
				for i in range(0,self.ps.queue.time.size):
					if self.ps.queue.time[i] > self.start_time:
						min_limit = i
						break
				self.init_position = self.ps.queue.position[:,0]
				self.verify_set = UpdateDataSet(self.verify_set,
									self.ps.queue,New,min_limit=min_limit)
				if not self.train_mod_set == None:
					self.verify_mod_set = None
				self.verify_set_size = self.ps.minimum
			else:
				if self.verified == False:
					raise ValueError('ModelDatabase verification set cannot\
									  be updated from an empty set')
				else:
					for i in range(0,self.ps.queue.time.size):
						if self.ps.queue.time[i] > self.start_time:
							min_limit = i
							break
					self.ps.queue.time = (self.ps.queue.time
										 + self.verify_set.time[-1])
					self.verify_set = UpdateDataSet(self.verify_set,
										self.ps.queue,New,min_limit=min_limit)
					if not self.train_mod_set == None:
						self.verify_mod_set = None
					self.verify_set_size = self.ps.minimum
		elif Set.lower() == "test":
			if New:
				if self.tested == True:
					print "Overwriting verification dataset"
				else:
					self.tested = True
				for i in range(0,self.ps.queue.time.size):
					if self.ps.queue.time[i] > self.start_time:
						min_limit = i
						break
				self.init_position = self.ps.queue.position[:,0]
				self.test_set = UpdateDataSet(self.test_set,
									self.ps.queue,New,min_limit=min_limit)
				if not self.train_mod_set == None:
					self.test_mod_set = None
				self.test_set_size = self.ps.minimum
				# print self.ps.count
				# print self.ps.minimum
			else:
				if self.tested == False:
					raise ValueError('ModelDatabase verification set cannot \
									  be updated from an empty set')
				else:
					for i in range(0,self.ps.queue.time.size):
						if self.ps.queue.time[i] > self.start_time:
							min_limit = i
							break
					self.ps.queue.time = (self.ps.queue.time
										  + self.test_set.time[-1])
					self.test_set = UpdateDataSet(self.test_set,
										self.ps.queue,New,min_limit=min_limit)
					if not self.train_mod_set == None:
						self.test_mod_set = None
					self.test_set_size = self.ps.minimum
		else:
			raise ValueError('Set name for UpdateSet was not defined \
							  properly')

	def downSample(self,Set):
		if Set.lower() == "train":
			datapoints = self.train_set.time.size
			orig_freq = float(datapoints)/(self.train_set.time[-1]
										   - self.train_set.time[0])
			print "Original Frequency: ", orig_freq

			if self.data_cap<datapoints:
				limited_f = float(self.data_cap)/float(datapoints)*orig_freq
				if limited_f<self.downsample_f:
					downsample_f = limited_f
					print "Data Cap Limitting"
				else:
					downsample_f = self.downsample_f
				print "Downsampled Frequency: ", downsample_f
			elif orig_freq < self.downsample_f:
				print "No Downsampling Required"
				return
			else:
				downsample_f = self.downsample_f
				print "Downsampled Frequency: ", downsample_f
			T = 1/downsample_f
			duration = self.train_set.time[-1]-self.train_set.time[0]
			start_time = self.train_set.time[0]
			N = int(math.ceil(duration/T))
			index = -1
			index_set = [0]
			for i in range(1,N):
				for j in range(index+1,self.train_set.time.size):
					if self.train_set.time[j]-start_time > i*T:
						index = j
						index_set.append(index)
						break
			self.train_mod_set = AverageDataSet(self.train_set,index_set)

		elif Set.lower() == "verify":
			datapoints = self.verify_set.time.size
			orig_freq = float(datapoints)/(self.verify_set.time[-1]
										   - self.verify_set.time[0])
			print "Original Frequency: ", orig_freq

			if self.data_cap<datapoints:
				limited_f = float(self.data_cap)/float(datapoints)*orig_freq
				if limited_f<self.downsample_f:
					downsample_f = limited_f
					print "Data Cap Limitting"
				else:
					downsample_f = self.downsample_f
				print "Downsampled Frequency: ", downsample_f
			elif orig_freq < self.downsample_f:
				print "No Downsampling Required"
				return
			else:
				downsample_f = self.downsample_f
				print "Downsampled Frequency: ", downsample_f
			T = 1/downsample_f
			duration = self.verify_set.time[-1]-self.verify_set.time[0]
			start_time = self.verify_set.time[0]
			N = int(math.ceil(duration/T))

			index = -1
			index_set = [0]
			for i in range(1,N):
				for j in range(index+1,self.verify_set.time.size):
					if self.verify_set.time[j]-start_time > i*T:
						index = j
						index_set.append(index)
						break
			self.verify_mod_set = AverageDataSet(self.verify_set,index_set)

		elif Set.lower() == "test":
			datapoints = self.test_set.time.size
			orig_freq = float(datapoints)/(self.test_set.time[-1]
										   - self.test_set.time[0])
			print "Original Frequency: ", orig_freq

			if self.data_cap<datapoints:
				limited_f = float(self.data_cap)/float(datapoints)*orig_freq
				if limited_f<self.downsample_f:
					downsample_f = limited_f
					print "Data Cap Limitting"
				else:
					downsample_f = self.downsample_f
				print "Downsampled Frequency: ", downsample_f
			elif orig_freq < self.downsample_f:
				print "No Downsampling Required"
				return
			else:
				downsample_f = self.downsample_f
				print "Downsampled Frequency: ", downsample_f
			T = 1/downsample_f
			duration = self.test_set.time[-1]-self.test_set.time[0]
			start_time = self.test_set.time[0]
			N = int(math.ceil(duration/T))

			index = -1
			index_set = [0]
			for i in range(1,N):
				for j in range(index+1,self.test_set.time.size):
					if self.test_set.time[j]-start_time > i*T:
						index = j
						index_set.append(index)
						break
			self.test_mod_set = AverageDataSet(self.test_set,index_set)

		else:
			raise ValueError("Downsample set could not be determined")

	def updateModel(self,optimize=False,restarts=0,gaus_noise=10):
		if self.train_mod_set == None:
			time = self.train_set.time
			position = self.train_set.position
			positionCmd = self.train_set.positionCmd
			velocity = self.train_set.velocityFlt
			velocityCmd = self.train_set.velocityCmd
			accel = self.train_set.accel
			accelCmd = self.train_set.accelCmd
			deflection = self.train_set.deflection
			deflection_vel = self.train_set.deflection_vel
			torque = self.train_set.torque
			torqueCmd = self.train_set.torqueCmd
			torqueID = self.train_set.torqueID
			temperature = self.train_set.motorSensorTemperature
		else:
			time = self.train_mod_set.time
			position = self.train_mod_set.position
			positionCmd = self.train_mod_set.positionCmd
			velocity = self.train_mod_set.velocityFlt
			velocityCmd = self.train_mod_set.velocityCmd
			accel = self.train_mod_set.accel
			accelCmd = self.train_mod_set.accelCmd
			deflection = self.train_mod_set.deflection
			deflection_vel = self.train_mod_set.deflection_vel
			torque = self.train_mod_set.torque
			torqueCmd = self.train_mod_set.torqueCmd
			torqueID = self.train_mod_set.torqueID
			temperature = self.train_mod_set.motorSensorTemperature
		train_position = position[self.joints_ML]
		train_positionCmd = positionCmd[self.joints_ML]
		train_velocity = velocity[self.joints_ML]
		train_velocityCmd = velocityCmd[self.joints_ML]
		train_accel = accel[self.joints_ML]
		train_accelCmd = accelCmd[self.joints_ML]
		train_deflection = deflection[self.joints_ML]
		train_deflection_vel = deflection_vel[self.joints_ML]
		train_torque = torque[self.joints_ML]
		train_torqueCmd = torqueCmd[self.joints_ML]
		train_torqueID = torqueID[self.joints_ML]
		train_temperature = temperature[self.joints_ML]

		if len(self.joints_ML) == 1:
			num_training_points = time.size
			train_position.shape = (num_training_points,1)
			train_positionCmd.shape = (num_training_points,1)
			train_velocity.shape = (num_training_points,1)
			train_velocityCmd.shape = (num_training_points,1)
			train_accel.shape = (num_training_points,1)
			train_accelCmd.shape = (num_training_points,1)
			train_deflection.shape = (num_training_points,1)
			train_deflection_vel.shape = (num_training_points,1)
			train_torque.shape = (num_training_points,1)
			train_torqueCmd.shape = (num_training_points,1)
			train_torqueID.shape = (num_training_points,1)
			train_temperature.shape = (num_training_points,1)
		else:
			train_position = np.transpose(train_position)
			train_positionCmd = np.transpose(train_positionCmd)
			train_velocity = np.transpose(train_velocity)
			train_velocityCmd = np.transpose(train_velocityCmd)
			train_accel = np.transpose(train_accel)
			train_accelCmd = np.transpose(train_accelCmd)
			train_deflection = np.transpose(train_deflection)
			train_deflection_vel = np.transpose(train_deflection_vel)
			train_torque = np.transpose(train_torque)
			train_torqueCmd = np.transpose(train_torqueCmd)
			train_torqueID = np.transpose(train_torqueID)
			train_temperature = np.transpose(train_temperature)

		train_eps_tau_centered = np.zeros(train_position.shape)
		eps_tau = train_torqueCmd-train_torqueID
		# print train_position.shape
		# print eps_tau.shape

		epsOffset = np.zeros((self.joints_ML.size,1))
		length_scale = []
		first_joint = 0
		joint_length_scale = [[40,0.2,5.,0.07,1.,1.],#40,0.25,12
							  [40,0.5,5.,1.,1.,1.],
							  [40,0.5,5.,10.,1.,1.],
							  [40,0.5,5.,70.,1,1.],
							  [0.7,20,5.,1.,1.,1.]]

		if self.joints_ML.size == 1:
			epsOffset = sum(eps_tau)/(eps_tau.size)
			train_eps_tau_centered = eps_tau - epsOffset
			length_scale.append(joint_length_scale[self.joints_ML][0]) #Position length scale for module
			length_scale.append(joint_length_scale[self.joints_ML][1]) #Velocity length scale
			length_scale.append(joint_length_scale[self.joints_ML][2]) #Acceleration length scale
			

			if self.learn_temp and self.learn_def:
				X = np.hstack((train_position,train_velocity,train_accel,train_deflection,train_temperature))
				length_scale.append(joint_length_scale[self.joints_ML][3]) #Deflection length scale
				length_scale.append(joint_length_scale[self.joints_ML][4]) #Temperature length scale
				dimen = 5
			elif self.learn_def:
				X = np.hstack((train_position,train_velocity,train_accel,train_deflection,train_deflection_vel))
				length_scale.append(joint_length_scale[self.joints_ML][3]) #Deflection length scale
				length_scale.append(joint_length_scale[self.joints_ML][4])
				dimen = 5
			elif self.learn_temp:
				X = np.hstack((train_position,train_velocity,train_accel,train_temperature))
				length_scale.append(joint_length_scale[self.joints_ML][4]) #Temperature length scale
				dimen = 4
			else:
				X = np.hstack((train_position,train_velocity,train_accel))
				dimen = 3


		else:
			for i in range(0,self.joints_ML.size):
				epsOffset[i] = sum(eps_tau[:,i])/(eps_tau[:,i].size)
				train_eps_tau_centered[:,i] = eps_tau[:,i] - epsOffset[i]
				length_scale.append(joint_length_scale[self.joints_ML[i]][0]) #Position length scale for module
				length_scale.append(joint_length_scale[self.joints_ML[i]][1]) #Velocity length scale
				length_scale.append(joint_length_scale[self.joints_ML[i]][2]) #Acceleration length scale

				if first_joint==0:
					if self.learn_temp and self.learn_def:
						X = np.vstack((train_position[:,i],train_velocity[:,i],train_accel[:,i],train_deflection[:,i],train_temperature[:,i]))
						length_scale.append(joint_length_scale[self.joints_ML[i]][3]) #Deflection length scale
						length_scale.append(joint_length_scale[self.joints_ML[i]][4]) #Temperature length scale
						dimen = 5
					elif self.learn_def:
						print train_deflection_vel[:,i].shape
						print train_deflection[:,i].shape
						X = np.vstack((train_position[:,i],train_velocity[:,i],train_accel[:,i],train_deflection[:,i],train_deflection_vel[:,i]))
						length_scale.append(joint_length_scale[self.joints_ML[i]][3]) #Deflection length scale
						length_scale.append(joint_length_scale[self.joints_ML[i]][4]) #Deflection velocity length scale
						dimen = 5
					elif self.learn_temp:
						X = np.vstack((train_position[:,i],train_velocity[:,i],train_accel[:,i],train_temperature[:,i]))
						length_scale.append(joint_length_scale[self.joints_ML[i]][4]) #Temperature length scale
						dimen = 4
					else:
						X = np.vstack((train_position[:,i],train_velocity[:,i],train_accel[:,i]))
						dimen = 3
					first_joint = 1
				else:
					if self.learn_temp and self.learn_def:
						X = np.vstack((X,train_position[:,i],train_velocity[:,i],train_accel[:,i],train_deflection[:,i],train_temperature[:,i]))
						length_scale.append(joint_length_scale[self.joints_ML[i]][3]) #Deflection length scale
						length_scale.append(joint_length_scale[self.joints_ML[i]][4]) #Temperature length scale
					elif self.learn_def:
						X = np.vstack((X,train_position[:,i],train_velocity[:,i],train_accel[:,i],train_deflection[:,i],train_deflection_vel[:,i]))
						length_scale.append(joint_length_scale[self.joints_ML[i]][3]) #Deflection length scale
						length_scale.append(joint_length_scale[self.joints_ML[i]][4]) #Deflection vel length scale
					elif self.learn_temp:
						X = np.vstack((X,train_position[:,i],train_velocity[:,i],train_accel[:,i],train_temperature[:,i]))
						length_scale.append(joint_length_scale[self.joints_ML[i]][3]) #Temperature length scale
					else:
						X = np.vstack((X,train_position[:,i],train_velocity[:,i],train_accel[:,i]))

		#Calling [:,i] makes the column vectors into row vectors above, hence the need for a transpose
		if len(self.joints_ML) == 1:
			train_eps_tau_centered.shape = (num_training_points,1)
		else:
			X = np.transpose(X)

		Y = train_eps_tau_centered

		var = 5
		ndim = dimen*self.joints_ML.size

		self.kernel = GPy.kern.RBF(input_dim=ndim,lengthscale=length_scale,variance=var,ARD=True)
		self.model = GPy.models.GPRegression(X,Y,self.kernel)
		self.model.Gaussian_noise = gaus_noise
		self.model.Gaussian_noise.variance.fix()
		print self.model
		print self.model.rbf.lengthscale
		if optimize:
			if restarts>0:
				self.model.optimize_restarts(num_restarts=restarts)
			else:
				self.model.optimize()
		print self.model
		print self.model.rbf.lengthscale
		self.epsOffset = epsOffset

		if len(self.joints_ML) == 1:
			train_position.shape = (1,num_training_points)
			train_positionCmd.shape = (1,num_training_points)
			train_velocity.shape = (1,num_training_points)
			train_velocityCmd.shape = (1,num_training_points)
			train_accel.shape = (1,num_training_points)
			train_accelCmd.shape = (1,num_training_points)
			train_deflection.shape = (1,num_training_points)
			train_deflection_vel.shape = (1,num_training_points)
			train_torque.shape = (1,num_training_points)
			train_torqueCmd.shape = (1,num_training_points)
			train_torqueID.shape = (1,num_training_points)
			train_temperature.shape = (1,num_training_points)

	def verifyData(self,Set="verify"):
		if Set.lower() == "verify":
			data = self.verify_set
		if Set.lower() == "train":
			data = self.train_set
		elif Set.lower() == "test":
			data = self.test_set

		time = data.time
		position = data.position
		positionCmd = data.positionCmd
		velocity = data.velocityFlt
		velocityCmd = data.velocityCmd
		accel = data.accel
		accelCmd = data.accelCmd
		deflection = data.deflection
		deflection_vel = data.deflection_vel
		torque = data.torque
		torqueCmd = data.torqueCmd
		torqueID = data.torqueID
		temperature = data.motorSensorTemperature

		verify_position = position[self.joints_ML]
		verify_positionCmd = positionCmd[self.joints_ML]
		verify_velocity = velocity[self.joints_ML]
		verify_velocityCmd = velocityCmd[self.joints_ML]
		verify_accel = accel[self.joints_ML]
		verify_accelCmd = accelCmd[self.joints_ML]
		verify_deflection = deflection[self.joints_ML]
		verify_deflection_vel = deflection_vel[self.joints_ML]
		verify_torque = torque[self.joints_ML]
		verify_torqueCmd = torqueCmd[self.joints_ML]
		verify_torqueID = torqueID[self.joints_ML]
		verify_temperature = temperature[self.joints_ML]

		if len(self.joints_ML) == 1:
			num_verifying_points = time.size
			verify_position.shape = (num_verifying_points,1)
			verify_positionCmd.shape = (num_verifying_points,1)
			verify_velocity.shape = (num_verifying_points,1)
			verify_velocityCmd.shape = (num_verifying_points,1)
			verify_accel.shape = (num_verifying_points,1)
			verify_accelCmd.shape = (num_verifying_points,1)
			verify_deflection.shape = (num_verifying_points,1)
			verify_deflection_vel.shape = (num_verifying_points,1)
			verify_torque.shape = (num_verifying_points,1)
			verify_torqueCmd.shape = (num_verifying_points,1)
			verify_torqueID.shape = (num_verifying_points,1)
			verify_temperature.shape = (num_verifying_points,1)
		else:
			verify_position = np.transpose(verify_position)
			verify_positionCmd = np.transpose(verify_positionCmd)
			verify_velocity = np.transpose(verify_velocity)
			verify_velocityCmd = np.transpose(verify_velocityCmd)
			verify_accel = np.transpose(verify_accel)
			verify_accelCmd = np.transpose(verify_accelCmd)
			verify_deflection = np.transpose(verify_deflection)
			verify_deflection_vel = np.transpose(verify_deflection_vel)
			verify_torque = np.transpose(verify_torque)
			verify_torqueCmd = np.transpose(verify_torqueCmd)
			verify_torqueID = np.transpose(verify_torqueID)
			verify_temperature = np.transpose(verify_temperature)

		if self.joints_ML.size == 1:
			if self.learn_temp and self.learn_def:
				X_test = np.hstack((verify_position,verify_velocity,verify_accel,verify_deflection,verify_temperature))
				dimen = 5
			elif self.learn_def:
				X_test = np.hstack((verify_position,verify_velocity,verify_accel,verify_deflection))
				dimen = 4
			elif self.learn_temp:
				X_test = np.hstack((verify_position,verify_velocity,verify_accel,verify_temperature))
				dimen = 4
			else:
				X_test = np.hstack((verify_position,verify_velocity,verify_accel))
				dimen = 3
		else:
			first_joint = 0
			for i in range(0,self.joints_ML.size):
				if first_joint==0:
					if self.learn_temp and self.learn_def:
						X_test = np.vstack((verify_position[:,i],verify_velocity[:,i],verify_accel[:,i],verify_deflection[:,i],verify_temperature[:,i]))
						dimen = 5
					elif self.learn_def:
						X_test = np.vstack((verify_position[:,i],verify_velocity[:,i],verify_accel[:,i],verify_deflection[:,i],verify_deflection_vel[:,i]))
						dimen = 5
					elif self.learn_temp:
						X_test = np.vstack((verify_position[:,i],verify_velocity[:,i],verify_accel[:,i],verify_temperature[:,i]))
						dimen = 4
					else:
						X_test = np.vstack((verify_position[:,i],verify_velocity[:,i],verify_accel[:,i]))
						dimen = 3
					first_joint = 1
				else:
					if self.learn_temp and self.learn_def:
						X_test = np.vstack((X_test,verify_position[:,i],verify_velocity[:,i],verify_accel[:,i],verify_deflection[:,i],verify_temperature[:,i]))
						dimen = 5
					elif self.learn_def:
						X_test = np.vstack((X_test,verify_position[:,i],verify_velocity[:,i],verify_accel[:,i],verify_deflection[:,i],verify_deflection_vel[:,i]))
						dimen = 5
					elif self.learn_temp:
						X_test = np.vstack((X_test,verify_position[:,i],verify_velocity[:,i],verify_accel[:,i],verify_temperature[:,i]))
						dimen = 4
					else:
						X_test = np.vstack((X_test,verify_position[:,i],verify_velocity[:,i],verify_accel[:,i]))
						dimen = 3
			X_test = X_test.T
		
		eps_pred, cov_pred = self.model.predict(X_test)
		eps_pred = eps_pred.T + self.epsOffset
		for i in range(len(self.joints_ML)):
			eps_a = verify_torqueCmd[:,i] - verify_torqueID[:,i]
			eps_p = eps_pred[i]
			verify_torqueGP = verify_torqueID[:,i]+eps_p

			target_variance = np.var(verify_torqueCmd[:,i])
			nMSE_RBD = np.mean(np.square(verify_torqueCmd[:,i]-verify_torqueID[:,i]))/target_variance
			nMSE_GP =  np.mean(np.square(verify_torqueCmd[:,i]-(verify_torqueID[:,i]+eps_p)))/target_variance

			percent_improvement = (nMSE_RBD-nMSE_GP)/nMSE_RBD*100

			print 'nMSE_RBD = ', nMSE_RBD
			print 'nMSE_RBD+GP = ', nMSE_GP
			print 'Percentage Improvement = ', percent_improvement

			# plt.figure(i)
			# plt.plot(verify_torqueCmd[:,i],'g',label='Reference Torque')
			# plt.plot(verify_torqueID[:,i],'b',label='RBD Torque')
			# if self.learn_temp and self.learn_def:
			# 	plt.plot(verify_torqueGP,'m',label='RBD+GP Torque (Def+Temp)')
			# elif self.learn_temp:
			# 	plt.plot(verify_torqueGP,'c',label='RBD+GP Torque (Temp)')
			# # elif self.learn_def:
			# 	# plt.plot(verify_torqueGP,'y',label='RBD+GP Torque (Def)')
			# else:
			# 	plt.plot(verify_torqueGP,'r',label='RBD+GP Torque')

			# plt.figure(i+10)
			# plt.plot(time,eps_a,'b',label='Actual Error')
			# if self.learn_temp and self.learn_def:
			# 	plt.plot(time,eps_p,'m',linewidth=2)
			# elif self.learn_temp:
			# 	plt.plot(time,eps_p,'c',linewidth=2)
			# # elif self.learn_def:
			# 	# plt.plot(time,eps_p,'y',linewidth=2)
			# else:
			# 	plt.plot(time,eps_p,'r',linewidth=2,label='Predicted Error')
			
			# # plt.title('Comparison of Actual Error Verus Model Predictive Error')
			# plt.ylabel('Torque [N-m]')
			# plt.xlabel('Time [s]')
			# plt.legend()

	def setSampleParams(self,downsample_f=20.,data_cap=2000):
		self.downsample_f = downsample_f
		self.data_cap = data_cap

	def setJointsToLearn(self,joints):
		self.joints_ML = joints

	def controller(self,motorOn,control_info):
		start_time = rospy.Time.now()
		time_from_start = 0.
		self.ps.final_time=control_info['tf']
		if self.ps.startBag:
			self.ps.bagging = True

		if control_info['type'] == "MinJerk":
				c_x = 0.0
				c_y = 0.3
				c_z = 0.1
				radius = 0.095
				waypoints = np.array([[-0.02, -0.02, -0.02, 0.365,  0.365, 0.365,   0.0,   0.0],
									  [ 0.49,  0.49,  0.49, 0.2175,  0.2175, 0.2175, 0.395, 0.395],
									  [  0.1,   -0.02,   0.1,  0.1,   -0.02,  0.1,   0.1,   0.1],
									  [   0.,   0.0,    0.,np.pi/2,np.pi/2,  0.0,   0.0,   0.0]])
				time_array = np.array([   2.,    2.,    2.,   2.,    2.,   2.,  2.,   2.])
				tf = np.sum(time_array)
				initial_angles= TG.inverseKinematics(c_x,c_y+radius,c_z,0)
		while control_info['tf']-time_from_start> 0:
			deflection = self.ps.queue.deflection[:,-1]
			cmd = hebi_cpp.msg.CommandML()
			trajectory = JointTrajectory()
			point = JointTrajectoryPoint()
			trajectory.header.stamp = rospy.Time.now()
			point.time_from_start = trajectory.header.stamp-start_time
			time_from_start = (point.time_from_start.secs
							  + point.time_from_start.nsecs/1000000000.)
			if control_info['type'] == "MinJerk":
				joint_const = TG.minJerkSetup_now(initial_angles,tf,waypoints,t_array=time_array)
				pos_v,vel_v,accel_v = TG.minJerkStep_now(time_from_start,tf,waypoints,joint_const,t_array=time_array)

			for joint in range(0,5):
				if joint == 0 or joint==1 or joint == 2 or joint == 3 or joint == 4:
					# if Set.lower() == "train":
					if control_info['type'] == "SuperSine":
						c_x = 0.0
						c_y = 0.3
						c_z = -0.025
						radius = 0.095
						initial_position = TG.inverseKinematics(c_x,c_y+radius,c_z,0)
						# initial_position = [1.68437, -0.445017, 1.18675, 4.7561,7.94885]
						pos,vel,accel = superpositionSine(time_from_start,
													amp=control_info['amp'][joint],
													f=control_info['freq'][joint],
													phi=control_info['phi'][joint])
						pos = pos+initial_position[joint]
					elif control_info['type'] == "Circle":
						pos_v,vel_v,accel_v = TG.generateJointTrajectory_now(time_from_start)
						pos = pos_v[joint]
						vel = vel_v[joint]
						accel = accel_v[joint]
					elif control_info['type'] == "MinJerk":
						pos = pos_v[joint]
						vel = vel_v[joint]
						accel = accel_v[joint]
				# elif joint == 1:
				# 	pos,vel,accel = superpositionSine(time_from_start,
				# 								amp=[np.pi/36,np.pi/32],
				# 								f=control_info['freq'])
				else:
					pos,vel,accel = 0.,0.,0.
				point.positions.append(pos)
				point.velocities.append(vel)
				point.accelerations.append(accel)
				 
			first_joint = 0
			eps = [0.,0.,0.,0.,0.]
			if control_info['ml']:
				color = std_msgs.msg.ColorRGBA();
				color.r = 0.0;
				color.g = 1.0;
				color.b = 0.0;
				color.a = 0.5;
				self.ps.path2_pub.publish(color)
				for i in self.joints_ML:
					if first_joint==0:
						if self.learn_temp and self.learn_def:
							X_test = np.vstack((point.positions[i],
												point.velocities[i],
												point.accelerations[i],
												deflection[i],temperature[i]))
							dimen = 5
						elif self.learn_def:
							X_test = np.vstack((point.positions[i],
												point.velocities[i],
												point.accelerations[i],
												deflection[i]))
							dimen = 4
						elif self.learn_temp:
							X_test = np.vstack((point.positions[i],
												point.velocities[i],
												point.accelerations[i],
												temperature[i]))
							dimen = 4
						else:
							X_test = np.vstack((point.positions[i],
												point.velocities[i],
												point.accelerations[i]))
							dimen = 3
						first_joint = 1
					else:
						if self.learn_temp and self.learn_def:
							X_test = np.vstack((X_test,point.positions[i],
												point.velocities[i],
												point.accelerations[i],
												deflection[i],temperature[i]))
						elif self.learn_def:
							X_test = np.vstack((X_test,point.positions[i],
												point.velocities[i],
												point.accelerations[i],
												deflection[i]))
						elif self.learn_temp:
							X_test = np.vstack((X_test,point.positions[i],
												point.velocities[i],
												point.accelerations[i],
												temperature[i]))
						else:
							X_test = np.vstack((X_test,point.positions[i],
												point.velocities[i],
												point.accelerations[i]))

				X_test.shape = (1,dimen*self.joints_ML.size)

				eps_pred,eps_cov = self.model.predict(X_test)
				for i in range(0,self.joints_ML.size):
					eps[self.joints_ML[i]] = eps_pred[0][i]+self.epsOffset[i]
			else:
				color = std_msgs.msg.ColorRGBA();
				color.r = 0.0;
				color.g = 0.0;
				color.b = 1.0;
				color.a = 0.5;
				self.ps.path1_pub.publish(color)

			cmd.epsTau = eps

			cmd.jointTrajectory.points.append(point)
			cmd.motorOn = float(motorOn)
			cmd.controlType = 1.
			cmd.closedLoop = control_info['closedLoop']
			cmd.feedforward = control_info['feedforward']
			cmd.pos_gain = control_info['p_gain']
			cmd.vel_gain = control_info['v_gain']
			self.ps.traj_pub.publish(cmd)
			# if self.ps.bagging:
			# 	message = hebi_cpp.msg.FeedbackML()
			# 	self.ps.bag.write("armcontroller_fbk",message)
			# if control_info['ml']:
				# time.sleep(0.005)
			# else:
				# time.sleep(0.01)
		if self.ps.startBag:
			self.ps.startBag = False
			self.ps.bagging = False
			time.sleep(0.001)
			self.ps.bag.close()

if __name__ == '__main__':
	# 0. = motor off but code runs
	# 1. = motor on with model learning commands
	# 2. = motor on without model learning commands
	try:
		motorOn = 2.

		initial_pose = Pose();

		initial_pose.position.x = 0.;
		initial_pose.position.y = 0.4;
		initial_pose.position.z = 0.21;
		initial_pose.orientation.x = 0.0;
		initial_pose.orientation.y = 1.0;
		initial_pose.orientation.z = 0.0;
		initial_pose.orientation.w = 0.0;

		check_reset = False
		rospy.init_node('model_learner', anonymous=True)

		cap = [3000]
		gaus_noise = [1.25]
		nMSE_RBD_array = []
		nMSE_GP_array = []
		nMSE_array = []
		control_info = {}
		num_of_runs = len(gaus_noise)
		for j in range(num_of_runs):
			amp_string = 'pi%18,pi%16'
			control_info['type'] = "MinJerk"
			control_info['amp'] = [[np.pi/8,np.pi/10],
								   [np.pi/16,np.pi/14],
								   [np.pi/12,np.pi/10],
								   [np.pi/10,np.pi/6],
								   [np.pi/8,np.pi/6]]

			control_info['freq'] = [[0.3,0.2],
									[0.3,0.2],
									[0.3,0.2],
									[0.3,0.2],
									[0.3,0.2]]

			control_info['phi'] = [[0,0],
									[-np.pi/2,-np.pi/2],
									[-np.pi/2,-np.pi/2],
									[0,0],
									[0,0]]

			control_info['p_gain'] = [25.,25.,20.,10.,3.];
  			control_info['v_gain'] = [0.1,0.1,0.1,0.1,0.1];

			control_info['tf'] = 16.
			control_info['Set'] = "train"
			control_info['closedLoop'] = True
			control_info['feedforward'] = True
			control_info['ml'] = False
			# if motorOn:
			# 	directory_name = control_info['type'] + '-Amp[' + amp_string + ']-Freq' + str(control_info['freq']) + str(datetime.today())
			# 	os.makedirs(directory_name)
			# 	ps = pubSub(bag=False,folderName=directory_name,control_info=control_info)
			# else:
			# 	ps = pubSub(bag=False,control_info=control_info)


			###### Training the Model ######
			
			ps = pubSub(bag=False,control_info=control_info)
			resetPosition(motorOn,ps)
			ps.reset()
			db = modelDatabase(ps)#,deflection=True)
			db.data_cap = cap[0]
			db.setJointsToLearn(np.array([0,1,2,3,4]))

			db.controller(motorOn,control_info)
			resetPosition(motorOn,ps)
			db.updateSet(New=True,Set="train")

			control_info['type'] = "MinJerk"

			# control_info['amp'] = [[np.pi/8,np.pi/10],
			# 					   [np.pi/16,np.pi/14],
			# 					   [np.pi/12,np.pi/10],
			# 					   [np.pi/10,np.pi/6],
			# 					   [np.pi/8,np.pi/6]]
			
			# control_info['freq'] = [[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3]]
			# ps = pubSub(bag=False,control_info=control_info)
			# ps.reset()

			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=False,Set="train")

			# #### BURN IN ####
			# ps1 = pubSub(bag=False,flush=True,control_info=control_info)
			# ps1.reset()

			# control_info['freq'] = [[0.7,0.5],
			# 						[0.7,0.5],
			# 						[0.7,0.5],
			# 						[0.7,0.5],
			# 						[0.7,0.5]]

			# control_info['tf'] = 30.
			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps1)

			# ### TRAIN AGAIN ####
			# ps = pubSub(bag=False,control_info=control_info)
			# resetPosition(motorOn,ps)
			# ps.reset()
			# db = modelDatabase(ps)#,deflection=True)
			# db.data_cap = cap[0]
			# db.setJointsToLearn(np.array([0,1,2,3,4]))

			# control_info['tf'] = 10.
			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=False,Set="train")

			# control_info['amp'] = [[np.pi/6,np.pi/8],
			# 					   [np.pi/16,np.pi/14],
			# 					   [np.pi/12,np.pi/10],
			# 					   [np.pi/10,np.pi/6],
			# 					   [np.pi/8,np.pi/6]]
			
			# control_info['freq'] = [[0.3,0.1],
			# 						[0.3,0.1],
			# 						[0.3,0.1],
			# 						[0.3,0.1],
			# 						[0.3,0.1]]
			# ps = pubSub(bag=False,control_info=control_info)
			# ps.reset()

			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=False,Set="train")
			# db.downSample(Set="train")



			
			#Create Deflection Model
			# db_def = modelDatabase(ps,deflection=True)
			# db_def.train_set = db.train_set
			# db_def.train_mod_set = db.train_mod_set

			# db_def.joints_ML = db.joints_ML

			#Update and potaentially optimize Models			
			db.updateModel(optimize=False,gaus_noise=gaus_noise[j])
			# db_def.updateModel(optimize=False,gaus_noise=gaus_noise[j])

			ps.unregister()

			######
			time.sleep(2)

			# control_info['feedforward'] = False
			# control_info['p_gain'] = [100.,120.,120.,20.,8.] #[60.,80.,80.,20.,8.];
  	# 		control_info['v_gain'] = [1.,1.,1.,0.1,0.1] #[1.,1.,1.,0.1,0.1]

  			control_info['Set'] = "verify"
			control_info['closedLoop'] = True
			control_info['feedforward'] = True
			control_info['ml'] = True
			control_info['freq'] = [[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3]]

			# control_info['p_gain'] = [14.,14.,14.,8.,2.];
  	# 		control_info['v_gain'] = [0.1,0.1,0.1,0.1,0.1];
			ps = pubSub(bag=False,control_info=control_info)
			db.ps = ps
			ps.reset()
			motorOn = 1.
			db.data_cap = cap[0]
			db.setJointsToLearn(np.array([0,1,2,3,4]))

			db.controller(motorOn,control_info)
			resetPosition(motorOn,ps)
			db.updateSet(New=True,Set="verify")

  			control_info['Set'] = "train"
			control_info['closedLoop'] = True
			control_info['feedforward'] = True
			control_info['ml'] = True
			control_info['freq'] = [[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3]]

			# control_info['p_gain'] = [14.,14.,14.,8.,2.];
  	# 		control_info['v_gain'] = [0.1,0.1,0.1,0.1,0.1];
			ps = pubSub(bag=False,control_info=control_info)
			db.ps = ps
			ps.reset()
			motorOn = 1.
			db.data_cap = cap[0]
			db.setJointsToLearn(np.array([0,1,2,3,4]))

			db.controller(motorOn,control_info)
			resetPosition(motorOn,ps)
			db.updateSet(New=False,Set="train")
			# db.updateSet(New=False,Set="train")

			db.updateModel(optimize=False,gaus_noise=gaus_noise[j])

			###### Verifying the Model ######
			# control_info['Set'] = "verify"
			# control_info['closedLoop'] = True
			# control_info['ml'] = False
			# # control_info['amp'] = [[np.pi/8,np.pi/10],
			# # 					   [np.pi/18,np.pi/16],
			# # 					   [np.pi/14,np.pi/12],
			# # 					   [np.pi/14,np.pi/10],
			# # 					   [np.pi/14,np.pi/10]]
			# control_info['freq'] = [[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3]]
			# control_info['tf'] = 10.

			# # if motorOn:
			# # 	ps = pubSub(bag=True,folderName=directory_name,control_info=control_info)
			# # else:
			# # 	ps = pubSub(bag=False,control_info=control_info)
			# ps = pubSub(bag=False,control_info=control_info)
			# db.ps = ps
			# ps.reset()
			
			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=True,Set="verify")
			# db_def.verify_set = db.verify_set
			# # db.verifyData(Set="verify")
			# # db_def.verifyData(Set="verify")
			######

			##### Testing the Closed Loop Model ######
			control_info['Set'] = "test"
			control_info['closedLoop'] = True
			control_info['feedforward'] = True
			control_info['ml'] = True
			control_info['freq'] = [[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3]]

			# control_info['p_gain'] = [14.,14.,14.,8.,2.];
  	# 		control_info['v_gain'] = [0.1,0.1,0.1,0.1,0.1];
			# if motorOn:
			# 	ps = pubSub(bag=True,folderName=directory_name,control_info=control_info)
			# else:
			# 	ps = pubSub(bag=False,control_info=control_info)
			ps = pubSub(bag=False,control_info=control_info)
			db.ps = ps
			ps.reset()
			motorOn = 1.
			db.controller(motorOn,control_info)
			resetPosition(motorOn,ps)
			db.updateSet(New=True,Set="test")
			# db.updateSet(New=False,Set="train")
			#####

			##### Verifying the Closed Loop Model ######
			# control_info['Set'] = "verify"
			# control_info['closedLoop'] = True
			# control_info['ml'] = False
			# control_info['freq'] = [[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3],
			# 						[0.4,0.3]]
			# # if motorOn:
			# # 	ps = pubSub(bag=True,folderName=directory_name,control_info=control_info)
			# # else:
			# # 	ps = pubSub(bag=False,control_info=control_info)
			# ps = pubSub(bag=False,control_info=control_info)
			# db.ps = ps
			# ps.reset()
			# motorOn = 2.
			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=True,Set="verify")
			#####


			# for j in range(db.train_mod_set.time.size):
			# 	if db.train_mod_set.time[j]>10.:
			# 		final_index = j
			# 		break

			# cross_point = -0.7

			# for i in range(db.test_set.positionCmd[2].size):
			# 	if db.test_set.positionCmd[2][i] > cross_point:
			# 		if np.abs(db.test_set.positionCmd[2][i] - 2.0) > np.abs(db.test_set.positionCmd[2][i-1] - 2.0):
			# 			time_cross_GP = db.test_set.time[i-1]
			# 		else:
			# 			time_cross_GP = db.test_set.time[i]
			# 		break

			# for i in range(db.verify_set.positionCmd[2].size):
			# 	if db.verify_set.positionCmd[2][i] > cross_point:
			# 		if np.abs(db.verify_set.positionCmd[2][i] - 2.0) > np.abs(db.verify_set.positionCmd[2][i-1] - 2.0):
			# 			time_cross_PD = db.verify_set.time[i-1]
			# 		else:
			# 			time_cross_PD = db.verify_set.time[i]
			# 		break

			# time_offset_PD = time_cross_GP-time_cross_PD
			# print time_cross_PD
			# print time_cross_GP
			# print time_offset_PD

			final_index = db.verify_set.time.size
			time.sleep(2)

			# time_offset_PD = np.mean(db.train_set.time[100:500]-db.verify_set.time[100:500])
			
			for k in range(db.joints_ML.size):
				plt.figure(30+k)
				
				plt.plot(db.train_set.time[:final_index],db.train_set.position[db.joints_ML[k],:final_index],linewidth=3,label='RBD',color='b')
				plt.plot(db.verify_set.time[:final_index],db.verify_set.position[db.joints_ML[k],:final_index],linewidth=3,label='Task-Based GP Trial 1',color='m')
				# plt.scatter(db.verify_set.time[:final_index],db.verify_set.position[db.joints_ML[k],:final_index],label='No Learning')
				# plt.figure(40+k)
				# plt.plot(db.verify_set.time[:final_index],db.verify_set.velocity[0,:final_index])
				# plt.scatter(db.verify_set.time[:final_index],db.verify_set.velocity[0,:final_index],label='No Learning')

			####### TESTING 

			# control_info['Set'] = "test"
			# control_info['closedLoop'] = False
			# control_info['ml'] = True
			
			# if motorOn:
			# 	ps = pubSub(bag=True,folderName=directory_name,control_info=control_info)
			# else:
			# 	ps = pubSub(bag=False,control_info=control_info)
			# db.ps = ps
			# ps.reset()
			
			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=True,Set="test")
			# db.verifyData(Set="test")

			# time.sleep(2)

			#######

			#### TRACKING PLOTS #####

			

			N = 10000
			
			t0 = db.test_set.time[0]
			tf = 10.-t0
			dt = tf/N

			initial_position = db.init_position[db.joints_ML]
			initial_position.shape = (initial_position.size,1)

			# time_offset_GP = np.mean(db.train_set.time[100:500]-db.test_set.time[100:500])
			
			for k in range(db.joints_ML.size):
				idx = db.joints_ML[k]
				pos,vel,accel = superpositionSine(t0,amp=control_info['amp'][idx],f=control_info['freq'][idx])

				positionCmd = pos+initial_position[idx]
				velocityCmd = vel
				accelCmd = accel
				timeCmd = np.array(t0)
				t = t0

				for i in range(N):
					t += dt
					pos,vel,accel = (superpositionSine(t,amp=control_info['amp'][idx],f=control_info['freq'][idx]))
					positionCmd = np.hstack((positionCmd,pos+initial_position[idx]))
					velocityCmd = np.hstack((velocityCmd,vel))
					accelCmd = np.hstack((accelCmd,accel))
					timeCmd = np.hstack((timeCmd,t))

				RMSE_PD_pos = np.sqrt(np.mean(np.square(db.verify_set.positionCmd[db.joints_ML[k],:final_index]-db.verify_set.position[db.joints_ML[k],:final_index])))
				RMSE_RBD_pos = np.sqrt(np.mean(np.square(db.train_set.positionCmd[db.joints_ML[k],:final_index]-db.train_set.position[db.joints_ML[k],:final_index])))
				RMSE_GP_pos = np.sqrt(np.mean(np.square(db.test_set.positionCmd[db.joints_ML[k]]-db.test_set.position[db.joints_ML[k]])))
				print k+1,"Pos_RMSE_RBD: ", RMSE_PD_pos
				print k+1,"Pos_RMSE_RBD: ", RMSE_RBD_pos
				print k+1,"Pos_RMSE_GP: ", RMSE_GP_pos
				print k+1,"Percentage Improvement in Position: ", (RMSE_RBD_pos-RMSE_GP_pos)/RMSE_RBD_pos*100
				plt.figure(30+k)
				plt.plot(db.test_set.time,db.test_set.position[db.joints_ML[k]],linewidth=3,color="green",label='Task-Based GP Trial 2')
				# plt.scatter(db.test_set.time,db.test_set.position[db.joints_ML[k]],color="green",label='Learning')
				# plt.plot(timeCmd,positionCmd,'r--',linewidth=3,label='Reference')
				#plt.plot(db.train_set.time,db.train_set.positionCmd[db.joints_ML[k]],'r--',linewidth=3,label='Desired')
				plt.plot(db.test_set.time,db.test_set.positionCmd[db.joints_ML[k]],'r--',linewidth=3,label='Desired')
				# plt.plot(db.verify_set.time+time_offset_PD,db.verify_set.positionCmd[db.joints_ML[k]],'m--',linewidth=3,label='Desired')
				# plt.plot(db.verify_set.time,db.verify_set.positionCmd[db.joints_ML[k]],'m--',linewidth=3,label='Desired')
				plt.title('Joint Position Tracking Comparison of Unlearned Versus Learned')
				plt.legend()
				plt.ylabel('Position [rad]')
				plt.xlabel('Time [s]')
			
				# plt.figure(40+k)
				# plt.plot(db.test_set.time,db.test_set.velocity[db.joints_ML[k]],color="green")
				# plt.scatter(db.test_set.time,db.test_set.velocity[db.joints_ML[k]],color="green",label='Learning')
				# plt.plot(timeCmd,velocityCmd,'r--',linewidth=3,label='Reference')
				# plt.title('Joint Velocity Tracking Comparison of Unlearned Versus Learned')
				# plt.legend()
				# plt.ylabel('Velocity [rad/s]')
				# plt.xlabel('Time [s]')

				plt.figure(50+k)
				# plt.plot(db.train_set.time,db.train_set.torqueCmd[db.joints_ML[k]],color="cyan",label='Commanded Torque')
				plt.plot(db.test_set.time,db.test_set.torqueCmd[db.joints_ML[k]],color="green",label='Commanded Torque')
				plt.plot(db.test_set.time,db.test_set.torqueID[db.joints_ML[k]],color="blue", label='RBD Torque')
				plt.plot(db.test_set.time,db.test_set.torqueID[db.joints_ML[k]]+db.test_set.epsTau[db.joints_ML[k]],color="red",label='GP Torque')
				plt.title('Comparison of Commanded Torque with RBD Computed Torque and GP+RBD Torque')
				plt.legend()
				plt.ylabel('Torque [N-m]')
				plt.xlabel('Time [s]')

			c_x = 0.0
			c_y = 0.3
			c_z = -0.025
			radius = 0.095
			tf = 2.5

			N = 1000
			tim = np.linspace(0,tf,N)
			ref_x = []
			ref_y = []
			for i in range(N):
				ref_x = np.append(ref_x,c_x + radius*np.cos(2*np.pi*tim[i]/tf))
				ref_y = np.append(ref_y,c_y + radius*np.sin(2*np.pi*tim[i]/tf))

			ref_z = c_z*np.ones((ref_x.shape))

			verify_position = db.verify_set.position.T
			T = TG.forwardKinematics(verify_position[0])

			verify_x = T[0][3]
			verify_y = T[1][3]
			verify_z = T[2][3]

			for i in range(1,db.verify_set.time.size):
				T = TG.forwardKinematics(verify_position[i])
				verify_x = np.append(verify_x,T[0][3])
				verify_y = np.append(verify_y,T[1][3])
				verify_z = np.append(verify_z,T[2][3])


			test_position = db.test_set.position.T
			T = TG.forwardKinematics(test_position[0])

			test_x = T[0][3]
			test_y = T[1][3]
			test_z = T[2][3]

			for i in range(1,db.test_set.time.size):
				T = TG.forwardKinematics(test_position[i])
				test_x = np.append(test_x,T[0][3])
				test_y = np.append(test_y,T[1][3])
				test_z = np.append(test_z,T[2][3])

			train_position = db.train_set.position.T
			T = TG.forwardKinematics(train_position[0])

			train_x = T[0][3]
			train_y = T[1][3]
			train_z = T[2][3]

			for i in range(1,db.train_set.time.size):
				T = TG.forwardKinematics(train_position[i])
				train_x = np.append(train_x,T[0][3])
				train_y = np.append(train_y,T[1][3])
				train_z = np.append(train_z,T[2][3])


			plt.figure(99)
			plt.plot(verify_x,verify_y,'m',linewidth=2,label='PD with Gravity Comp')
			plt.plot(train_x,train_y,'b',linewidth=2,label='RBD')
			plt.plot(test_x,test_y,'g',linewidth=2,label='Task-Based GP')

			plt.figure(100)
			plt.plot(verify_x,verify_z,'m',linewidth=2,label='PD with Gravity Comp')
			plt.plot(train_x,train_z,'b',linewidth=2,label='RBD')
			plt.plot(test_x,test_z,'g',linewidth=2,label='Task-Based GP')

			## Relearn
			# db.updateModel(optimize=True,gaus_noise=gaus_noise[j])

			##### Testing the Closed Loop Model ######
			control_info['Set'] = "test"
			control_info['closedLoop'] = True
			control_info['ml'] = True
			control_info['freq'] = [[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3],
									[0.4,0.3]]
			# # if motorOn:
			# # 	ps = pubSub(bag=True,folderName=directory_name,control_info=control_info)
			# # else:
			# # 	ps = pubSub(bag=False,control_info=control_info)
			# ps = pubSub(bag=False,control_info=control_info)
			# db.ps = ps
			# ps.reset()
			# motorOn = 1.
			# db.controller(motorOn,control_info)
			# resetPosition(motorOn,ps)
			# db.updateSet(New=True,Set="test")
			#####

			N = 10000
			
			t0 = db.test_set.time[0]
			tf = 10.-t0
			dt = tf/N

			initial_position = db.init_position[db.joints_ML]
			initial_position.shape = (initial_position.size,1)

			
			# for k in range(db.joints_ML.size):
			# 	idx = db.joints_ML[k]
			# 	pos,vel,accel = superpositionSine(t0,amp=control_info['amp'][idx],f=control_info['freq'][idx])

			# 	positionCmd = pos+initial_position[idx]
			# 	velocityCmd = vel
			# 	accelCmd = accel
			# 	timeCmd = np.array(t0)
			# 	t = t0

			# 	for i in range(N):
			# 		t += dt
			# 		pos,vel,accel = (superpositionSine(t,amp=control_info['amp'][idx],f=control_info['freq'][idx]))
			# 		positionCmd = np.hstack((positionCmd,pos+initial_position[idx]))
			# 		velocityCmd = np.hstack((velocityCmd,vel))
			# 		accelCmd = np.hstack((accelCmd,accel))
			# 		timeCmd = np.hstack((timeCmd,t))

			# 	RMSE_RBD_pos = np.sqrt(np.mean(np.square(db.verify_set.positionCmd[db.joints_ML[k],:final_index]-db.verify_set.position[db.joints_ML[k],:final_index])))
			# 	RMSE_GP_pos = np.sqrt(np.mean(np.square(db.test_set.positionCmd[db.joints_ML[k]]-db.test_set.position[db.joints_ML[k]])))
			# 	print k+1,"Pos_RMSE_RBD: ", RMSE_RBD_pos
			# 	print k+1,"Pos_RMSE_GP: ", RMSE_GP_pos
			# 	print k+1,"Percentage Improvement in Position: ", (RMSE_RBD_pos-RMSE_GP_pos)/RMSE_RBD_pos*100

			# 	plt.figure(30+k)
			# 	plt.plot(db.test_set.time,db.test_set.position[db.joints_ML[k]],color="cyan")
			# 	#plt.scatter(db.test_set.time,db.test_set.position[db.joints_ML[k]],color="cyan",label='ReLearning')
			# 	#plt.plot(timeCmd,positionCmd,'r--',linewidth=3,label='Reference')
			# 	#plt.plot(db.verify_set.time,db.verify_set.positionCmd[db.joints_ML[k]],'b--',linewidth=3,label='Reference')
			# 	plt.plot(db.test_set.time,db.test_set.positionCmd[db.joints_ML[k]],'c--',linewidth=3,label='Reference')
			# 	plt.title('Joint Position Tracking Comparison of Unlearned Versus Learned')
			# 	plt.legend()
			# 	plt.ylabel('Position [rad]')
			# 	plt.xlabel('Time [s]')
			
			# 	plt.figure(40+k)
			# 	plt.plot(db.test_set.time,db.test_set.velocity[db.joints_ML[k]],color="green")
			# 	plt.scatter(db.test_set.time,db.test_set.velocity[db.joints_ML[k]],color="green",label='Learning')
			# 	plt.plot(timeCmd,velocityCmd,'r--',linewidth=3,label='Reference')
			# 	plt.title('Joint Velocity Tracking Comparison of Unlearned Versus Learned')
			# 	plt.legend()
			# 	plt.ylabel('Velocity [rad/s]')
			# 	plt.xlabel('Time [s]')

			# 	plt.figure(50+k)
			# 	# plt.plot(db.train_set.time,db.train_set.torqueCmd[db.joints_ML[k]],color="cyan",label='Commanded Torque')
			# 	plt.plot(db.test_set.time,db.test_set.torqueCmd[db.joints_ML[k]],color="cyan",label='Commanded Torque')
			# 	#plt.plot(db.test_set.time,db.test_set.torqueID[db.joints_ML[k]],color="blue", label='RBD Torque')
			# 	#plt.plot(db.test_set.time,db.test_set.torqueID[db.joints_ML[k]]+db.test_set.epsTau[db.joints_ML[k]],color="red",label='GP Torque')
			# 	plt.title('Comparison of Commanded Torque with RBD Computed Torque and GP+RBD Torque')
			# 	plt.legend()
			# 	plt.ylabel('Torque [N-m]')
			# 	plt.xlabel('Time [s]')

			

			

			test_position = db.test_set.position.T
			T = TG.forwardKinematics(test_position[0])

			test_x = T[0][3]
			test_y = T[1][3]
			test_z = T[2][3]

			for i in range(1,db.test_set.time.size):
				T = TG.forwardKinematics(test_position[i])
				test_x = np.append(test_x,T[0][3])
				test_y = np.append(test_y,T[1][3])
				test_z = np.append(test_z,T[2][3])

			plt.figure(99)
			plt.plot(ref_x,ref_y,'r--',linewidth=2,label='Desired')
			# .plot(test_x,test_y,'c',linewidth=2,label='With Relearning')
			plt.axis('equal')
			plt.legend()
			plt.xlabel('X [m]')
			plt.ylabel('Y [m]')

			plt.figure(100)
			plt.plot(ref_x,ref_z,'r--',linewidth=2,label='Desired')
			# .plot(test_x,test_y,'c',linewidth=2,label='With Relearning')
			plt.axis('equal')
			plt.legend()
			plt.xlabel('X [m]')
			plt.ylabel('Z [m]')

			#### TRACKING PLOTS #####
			plt.show()
			

			#### COMMENTED IMPORTANT SECTION
				# cmd_variance = np.var(db.test_set.torqueCmd[db.joints_ML[k]])
				# nMSE_RBD = np.mean(np.square(db.test_set.torqueCmd[db.joints_ML[k]]-db.test_set.torqueID[db.joints_ML[k]]))/cmd_variance
				# nMSE_GP = np.mean(np.square(db.test_set.torqueCmd[db.joints_ML[k]]-(db.test_set.torqueID[db.joints_ML[k]]+db.test_set.epsTau[db.joints_ML[k]])))/cmd_variance

				# # for j in range(db.verify_set.time.size):
				# # 	if db.verify_set.time[j]>10.:
				# # 		final_index = j
				# # 		break
				
				# print k+1,"nMSE_RBD: ", nMSE_RBD
				# print k+1,"nMSE_GP: ", nMSE_GP
				# print k+1,"Percentage Improvement: ", (nMSE_RBD-nMSE_GP)/nMSE_RBD*100
				
				# nMSE_RBD_array.append(nMSE_RBD)
				# nMSE_GP_array.append(nMSE_GP)
				# nMSE_array.append((nMSE_RBD-nMSE_GP)/nMSE_RBD*100)


				

					# RMSE_RBD_vel = np.sqrt(np.mean(np.square(db.verify_set.velocityCmd[db.joints_ML[k],:final_index]-db.verify_set.velocity[db.joints_ML[k],:final_index])))
					# RMSE_GP_vel = np.sqrt(np.mean(np.square(db.test_set.velocityCmd[db.joints_ML[k]]-db.test_set.velocity[db.joints_ML[k]])))
					# print k+1,"Vel_RMSE_RBD: ", RMSE_RBD_vel
					# print k+1,"Vel_RMSE_GP :", RMSE_GP_vel
					# print k+1,"Percentage Improvement in Velocity: ", (RMSE_RBD_vel-RMSE_GP_vel)/RMSE_RBD_vel*100
			
				# print j/float(num_of_runs)*100., " Percent Done"


				#### COMMENTED IMPORTANT SECTION































































	# print nMSE_RBD_array
	# print nMSE_GP_array
	# print nMSE_array
	# plt.plot(cap,nMSE_array)
	# plt.title('Percent Improvement on Predicted Torque nMSE Over Data Points in Training Set')
	# plt.ylabel('Percent Improvement [%]')
	# plt.xlabel('Data Points in Training Set')

	# fig, ax = plt.subplots()
	# index = np.arange(len(cap))
	# bar_width = 0.25
	# opacity = 0.5
	 
	# rects1 = plt.bar(index, nMSE_RBD_array, bar_width,
	#					alpha=opacity,
	#					color='b',
	#					label='RBD Model')
	 
	# rects2 = plt.bar(index + bar_width, nMSE_GP_array, bar_width,
	#					alpha=opacity,
	#					color='g',
	#					abel='RBD+GP Model')
	 
	# plt.title('Prediction nMSE Over DataPoints in Training Set')
	# plt.ylabel('Torque nMSE')
	# plt.xlabel('Data Points in Training Set')
	# plt.xticks(index + bar_width, cap)
	# plt.legend()
	 
	# plt.tight_layout()



	# plt.figure()
	# plt.plot(db.train_mod_set.time,db.train_mod_set.torqueCmd[0],color="cyan")
	# plt.scatter(db.verify_set.time,db.verify_set.torqueID[0])
	# # plt.scatter(db.train_mod_set.time,db.train_mod_set.torqueID[0],color="red")
	# plt.plot(db.test_set.time,db.test_set.torqueCmd[0],color="red")
	# plt.scatter(db.test_set.time,db.test_set.torqueID[0],color="green")

	# plt.figure()
	# plt.scatter(db1.verify_set.time,db1.verify_set.position[0],color="cyan")
	# plt.scatter(db1.train_mod_set.time,db1.train_mod_set.position[0])
	# plt.title('Full Sample Frequency versus Downsampled Frequency of Data Points')
	# plt.ylabel('Position [rad]')
	# plt.xlabel('Time [s]')

	# plt.figure()
	# plt.scatter(db.verify_set.time,db.verify_set.position[0],color="cyan")
	# plt.scatter(db.train_mod_set.time,db.train_mod_set.position[0])
	# plt.title('Full Sample Frequency versus Downsampled Frequency of Data Points')
	# plt.ylabel('Position [rad]')
	# plt.xlabel('Time [s]')
	
	except:
		ps = pubSub()
		resetPosition(motorOn,ps)
		raise