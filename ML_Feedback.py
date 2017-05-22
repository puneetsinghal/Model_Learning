#!/usr/bin/env python
import rospy
import sys
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import std_msgs.msg

import argparse
from collections import namedtuple
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

import model_learning.msg

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

def resetPosition(motorOn,ps,position):
	tf = 4.
	start_time = rospy.Time.now()
	time_from_start = 0.

	cmd = model_learning.msg.CommandML()
	point = JointTrajectoryPoint()

	point.positions = TG.inverseKinematics(position.c_x,position.c_y,position.c_z,position.theta)
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
	#Publisher and subscriber class for communicating with the C++ arm_controller
	#via custom ROS messages (FeedbackML and CommandML)
	def __init__(self,bag=False,folderName=None,flush=False,control_info={}):
		#Initialization function for the publisher and subscriber class
		#	bag [in]	= specify if bagging of the data is desired
		#	folderName [in]	= name of the folder for saving the bag files
		#	flush [in]	=
		self.queue = dataStruct()
		self.restart_arm = True	#Clears the queue data and restarts collection
		self.count = 1
		self.minimum = 0
		self.initial_time = 0.
		self.prev_time = 0.
		self.bagging = False #Signals that bagging of the data is desired
		self.startBag = False #Signal to start collecting data in the bag while flag active
		self.flush = flush #Flush clears the queue by running the subscriber callback but not
						   #actually putting any of the datat into the queue

		#Create a ROS publisher for sending command signals to the arm controller
		self.traj_pub = rospy.Publisher("ml_publisher",
										model_learning.msg.CommandML,queue_size=1)
		#Create two ROS publishers for displaying end effector colors in Rviz
		self.path1_pub = rospy.Publisher("/path1/color",
							std_msgs.msg.ColorRGBA,queue_size=1)
		self.path2_pub = rospy.Publisher("/path2/color",
							std_msgs.msg.ColorRGBA,queue_size=1)
		#Create a ROS subscriber for collecting the arm controller feedback into a queue to be potentially
		#input into one of the modeldatabase datasets
		#Queue size is large since Python is substantially slower at processing the data than the arm
		#controller
		self.fbk_sub = rospy.Subscriber("armcontroller_fbk",
							model_learning.msg.FeedbackML,self.armControlCallback,
							queue_size=5000)
		#If the bag flag is active, create a rosbag file to save the incoming data
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
		
	def armControlCallback(self,ros_data):
		#Callback function for the armcontroller_fbk subscriber for every new feedback
		#message received
		#	ros_data[in]	= FeedbckML custom ROS message
		if self.restart_arm: #Reset the arm_controller queue
			self.addToQueue(ros_data,New=True)
			self.restart_arm = False
			self.count = 1
		elif not self.flush: #If flush is active, the update step is skipped entirely
			self.addToQueue(ros_data,New=False)
			self.count += 1
		self.minimum = self.count #Count the size of the queue
		if self.bagging:
			self.bag.write(control_info['Set'],ros_data)
			
	def addToQueue(self,fbk,New=True):
		#Add the current data into the queue
		#	fbk [in]	= FeedbackML custom ROS message
		#	New [in] 	= flag for designating if the queue is new
		if New: #Inititialize all of the arrays in the queue
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
		else: #Stack the new data point into the existent queue
			time_secs = float(fbk.header.stamp.secs)+float(fbk.header.stamp.nsecs)/1000000000. - self.initial_time
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
			
	def unregister(self):
		#Unregister the subscriber from the ROS topic
		self.fbk_sub.unregister()
		self.fbk_sub = None

	def reset(self):
		#Resets the publisher and subscriber class queue
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
		self.minimum_f = 50.
		self.start_time = 0.
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
				if limited_f<self.minimum_f:
					downsample_f = self.minimum_f
					print "Minimum",downsample_f,"Frequency Reached"
				elif limited_f<self.downsample_f:
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
				if limited_f<self.minimum_f:
					downsample_f = self.minimum_f
					print "Minimum",downsample_f,"Frequency Reached"
				elif limited_f<self.downsample_f:
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
				if limited_f<self.minimum_f:
					downsample_f = self.minimum_f
					print "Minimum",downsample_f,"Frequency Reached"
				elif limited_f<self.downsample_f:
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
		# print self.model
		# print self.model.rbf.lengthscale
		if optimize:
			if restarts>0:
				self.model.optimize_restarts(num_restarts=restarts)
			else:
				self.model.optimize()
		# print self.model
		# print self.model.rbf.lengthscale
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
			cmd = model_learning.msg.CommandML()
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
			# 	message = model_learning.msg.FeedbackML()
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

	def saveDatabase(self,train_label='train',test_label='test',
						verify_label ='verify'):
		#Save the database files to a csv, which is easier to work with
		#than a rosbag
		#	train_label [in]  = label to change the name of the saved train_set
		#	verify_label [in] = label to change the name of the saved verify_set
		#	test_label [in]   = label to change the name of the saved test_set
		train_label = train_label + '.csv'
		test_label = test_label + '.csv'
		verify_label = verify_label + '.csv'

		#Check if the database has a training set
		if hasattr(self,'train_set'):
			#Open a csvfile to write to
			with open(train_label, 'wb') as csvfile:
				#Create a csv writer with comma delimitation
				writer = csv.writer(csvfile, delimiter=',',
				                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
				#Create the column headers for all of the feedback data
				writer.writerow(['time',
				'position1','position2','position3','position4','position5',
				'positionCmd1','positionCmd2','positionCmd3','positionCmd4','positionCmd5',
				'velocity1','velocity2','velocity3','velocity4','velocity5',
				'velocityCmd1','velocityCmd2','velocityCmd3','velocityCmd4','velocityCmd5',
				'velocityFlt1','velocityFlt2','velocityFlt3','velocityFlt4','velocityFlt5',
				'accel1','accel2','accel3','accel4','accel5',
				'accelCmd1','accelCmd2','accelCmd3','accelCmd4','accelCmd5',
				'torque1','torque2','torque3','torque4','torque5',
				'torqueCmd1','torqueCmd2','torqueCmd3','torqueCmd4','torqueCmd5',
				'torqueID1','torqueID2','torqueID3','torqueID4','torqueID5',
				'deflection1','deflection2','deflection3','deflection4','deflection5',
				'deflection_vel1','deflection_vel2','deflection_vel3','deflection_vel4','deflection_vel5',
				'motorSensorTemperature1','motorSensorTemperature2','motorSensorTemperature3','motorSensorTemperature4','motorSensorTemperature5',
				'windingTemp1','windingTemp2','windingTemp3','windingTemp4','windingTemp5',
				'windingTempFlt1','windingTempFlt2','windingTempFlt3','windingTempFlt4','windingTempFlt5',
				'epsTau1','epsTau2','epsTau3','epsTau4','epsTau5'])

				#TODO: Find a way to iterate over values instead of calling each individually
				#TODO: Make into a separate function so it does not take up 3x the space
				#Add all of the data into the respecitve column
				for i in range(self.train_set.time.size):
					writer.writerow([self.train_set.time[i],
						self.train_set.position[0,i],
						self.train_set.position[1,i],
						self.train_set.position[2,i],
						self.train_set.position[3,i],
						self.train_set.position[4,i],
						self.train_set.positionCmd[0,i],
						self.train_set.positionCmd[1,i],
						self.train_set.positionCmd[2,i],
						self.train_set.positionCmd[3,i],
						self.train_set.positionCmd[4,i],
						self.train_set.velocity[0,i],
						self.train_set.velocity[1,i],
						self.train_set.velocity[2,i],
						self.train_set.velocity[3,i],
						self.train_set.velocity[4,i],
						self.train_set.velocityCmd[0,i],
						self.train_set.velocityCmd[1,i],
						self.train_set.velocityCmd[2,i],
						self.train_set.velocityCmd[3,i],
						self.train_set.velocityCmd[4,i],
						self.train_set.velocityFlt[0,i],
						self.train_set.velocityFlt[1,i],
						self.train_set.velocityFlt[2,i],
						self.train_set.velocityFlt[3,i],
						self.train_set.velocityFlt[4,i],
						self.train_set.accel[0,i],
						self.train_set.accel[1,i],
						self.train_set.accel[2,i],
						self.train_set.accel[3,i],
						self.train_set.accel[4,i],
						self.train_set.accelCmd[0,i],
						self.train_set.accelCmd[1,i],
						self.train_set.accelCmd[2,i],
						self.train_set.accelCmd[3,i],
						self.train_set.accelCmd[4,i],
						self.train_set.torque[0,i],
						self.train_set.torque[1,i],
						self.train_set.torque[2,i],
						self.train_set.torque[3,i],
						self.train_set.torque[4,i],
						self.train_set.torqueCmd[0,i],
						self.train_set.torqueCmd[1,i],
						self.train_set.torqueCmd[2,i],
						self.train_set.torqueCmd[3,i],
						self.train_set.torqueCmd[4,i],
						self.train_set.torqueID[0,i],
						self.train_set.torqueID[1,i],
						self.train_set.torqueID[2,i],
						self.train_set.torqueID[3,i],
						self.train_set.torqueID[4,i],
						self.train_set.deflection[0,i],
						self.train_set.deflection[1,i],
						self.train_set.deflection[2,i],
						self.train_set.deflection[3,i],
						self.train_set.deflection[4,i],
						self.train_set.deflection_vel[0,i],
						self.train_set.deflection_vel[1,i],
						self.train_set.deflection_vel[2,i],
						self.train_set.deflection_vel[3,i],
						self.train_set.deflection_vel[4,i],
						self.train_set.motorSensorTemperature[0,i],
						self.train_set.motorSensorTemperature[1,i],
						self.train_set.motorSensorTemperature[2,i],
						self.train_set.motorSensorTemperature[3,i],
						self.train_set.motorSensorTemperature[4,i],
						self.train_set.windingTemp[0,i],
						self.train_set.windingTemp[1,i],
						self.train_set.windingTemp[2,i],
						self.train_set.windingTemp[3,i],
						self.train_set.windingTemp[4,i],
						self.train_set.windingTempFlt[0,i],
						self.train_set.windingTempFlt[1,i],
						self.train_set.windingTempFlt[2,i],
						self.train_set.windingTempFlt[3,i],
						self.train_set.windingTempFlt[4,i],
						self.train_set.epsTau[0,i],
						self.train_set.epsTau[1,i],
						self.train_set.epsTau[2,i],
						self.train_set.epsTau[3,i],
						self.train_set.epsTau[4,i]])

		#Check if the database has a training set
		if hasattr(self,'test_set'):
			#Open a csvfile to write to
			with open(test_label, 'wb') as csvfile:
				#Create a csv writer with comma delimitation
				writer = csv.writer(csvfile, delimiter=',',
				                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
				#Create the column headers for all of the feedback data
				writer.writerow(['time',
				'position1','position2','position3','position4','position5',
				'positionCmd1','positionCmd2','positionCmd3','positionCmd4','positionCmd5',
				'velocity1','velocity2','velocity3','velocity4','velocity5',
				'velocityCmd1','velocityCmd2','velocityCmd3','velocityCmd4','velocityCmd5',
				'velocityFlt1','velocityFlt2','velocityFlt3','velocityFlt4','velocityFlt5',
				'accel1','accel2','accel3','accel4','accel5',
				'accelCmd1','accelCmd2','accelCmd3','accelCmd4','accelCmd5',
				'torque1','torque2','torque3','torque4','torque5',
				'torqueCmd1','torqueCmd2','torqueCmd3','torqueCmd4','torqueCmd5',
				'torqueID1','torqueID2','torqueID3','torqueID4','torqueID5',
				'deflection1','deflection2','deflection3','deflection4','deflection5',
				'deflection_vel1','deflection_vel2','deflection_vel3','deflection_vel4','deflection_vel5',
				'motorSensorTemperature1','motorSensorTemperature2','motorSensorTemperature3','motorSensorTemperature4','motorSensorTemperature5',
				'windingTemp1','windingTemp2','windingTemp3','windingTemp4','windingTemp5',
				'windingTempFlt1','windingTempFlt2','windingTempFlt3','windingTempFlt4','windingTempFlt5',
				'epsTau1','epsTau2','epsTau3','epsTau4','epsTau5'])

				#Add all of the data into the respecitve column
				for i in range(self.test_set.time.size):
					writer.writerow([self.test_set.time[i],
						self.test_set.position[0,i],
						self.test_set.position[1,i],
						self.test_set.position[2,i],
						self.test_set.position[3,i],
						self.test_set.position[4,i],
						self.test_set.positionCmd[0,i],
						self.test_set.positionCmd[1,i],
						self.test_set.positionCmd[2,i],
						self.test_set.positionCmd[3,i],
						self.test_set.positionCmd[4,i],
						self.test_set.velocity[0,i],
						self.test_set.velocity[1,i],
						self.test_set.velocity[2,i],
						self.test_set.velocity[3,i],
						self.test_set.velocity[4,i],
						self.test_set.velocityCmd[0,i],
						self.test_set.velocityCmd[1,i],
						self.test_set.velocityCmd[2,i],
						self.test_set.velocityCmd[3,i],
						self.test_set.velocityCmd[4,i],
						self.test_set.velocityFlt[0,i],
						self.test_set.velocityFlt[1,i],
						self.test_set.velocityFlt[2,i],
						self.test_set.velocityFlt[3,i],
						self.test_set.velocityFlt[4,i],
						self.test_set.accel[0,i],
						self.test_set.accel[1,i],
						self.test_set.accel[2,i],
						self.test_set.accel[3,i],
						self.test_set.accel[4,i],
						self.test_set.accelCmd[0,i],
						self.test_set.accelCmd[1,i],
						self.test_set.accelCmd[2,i],
						self.test_set.accelCmd[3,i],
						self.test_set.accelCmd[4,i],
						self.test_set.torque[0,i],
						self.test_set.torque[1,i],
						self.test_set.torque[2,i],
						self.test_set.torque[3,i],
						self.test_set.torque[4,i],
						self.test_set.torqueCmd[0,i],
						self.test_set.torqueCmd[1,i],
						self.test_set.torqueCmd[2,i],
						self.test_set.torqueCmd[3,i],
						self.test_set.torqueCmd[4,i],
						self.test_set.torqueID[0,i],
						self.test_set.torqueID[1,i],
						self.test_set.torqueID[2,i],
						self.test_set.torqueID[3,i],
						self.test_set.torqueID[4,i],
						self.test_set.deflection[0,i],
						self.test_set.deflection[1,i],
						self.test_set.deflection[2,i],
						self.test_set.deflection[3,i],
						self.test_set.deflection[4,i],
						self.test_set.deflection_vel[0,i],
						self.test_set.deflection_vel[1,i],
						self.test_set.deflection_vel[2,i],
						self.test_set.deflection_vel[3,i],
						self.test_set.deflection_vel[4,i],
						self.test_set.motorSensorTemperature[0,i],
						self.test_set.motorSensorTemperature[1,i],
						self.test_set.motorSensorTemperature[2,i],
						self.test_set.motorSensorTemperature[3,i],
						self.test_set.motorSensorTemperature[4,i],
						self.test_set.windingTemp[0,i],
						self.test_set.windingTemp[1,i],
						self.test_set.windingTemp[2,i],
						self.test_set.windingTemp[3,i],
						self.test_set.windingTemp[4,i],
						self.test_set.windingTempFlt[0,i],
						self.test_set.windingTempFlt[1,i],
						self.test_set.windingTempFlt[2,i],
						self.test_set.windingTempFlt[3,i],
						self.test_set.windingTempFlt[4,i],
						self.test_set.epsTau[0,i],
						self.test_set.epsTau[1,i],
						self.test_set.epsTau[2,i],
						self.test_set.epsTau[3,i],
						self.test_set.epsTau[4,i]])

		#Check if the database has a training set
		if hasattr(self,'verify_set'):
			#Open a csvfile to write to
			with open(verify_label, 'wb') as csvfile:
				#Create a csv writer with comma delimitation
				writer = csv.writer(csvfile, delimiter=',',
				                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
				#Create the column headers for all of the feedback data
				writer.writerow(['time',
				'position1','position2','position3','position4','position5',
				'positionCmd1','positionCmd2','positionCmd3','positionCmd4','positionCmd5',
				'velocity1','velocity2','velocity3','velocity4','velocity5',
				'velocityCmd1','velocityCmd2','velocityCmd3','velocityCmd4','velocityCmd5',
				'velocityFlt1','velocityFlt2','velocityFlt3','velocityFlt4','velocityFlt5',
				'accel1','accel2','accel3','accel4','accel5',
				'accelCmd1','accelCmd2','accelCmd3','accelCmd4','accelCmd5',
				'torque1','torque2','torque3','torque4','torque5',
				'torqueCmd1','torqueCmd2','torqueCmd3','torqueCmd4','torqueCmd5',
				'torqueID1','torqueID2','torqueID3','torqueID4','torqueID5',
				'deflection1','deflection2','deflection3','deflection4','deflection5',
				'deflection_vel1','deflection_vel2','deflection_vel3','deflection_vel4','deflection_vel5',
				'motorSensorTemperature1','motorSensorTemperature2','motorSensorTemperature3','motorSensorTemperature4','motorSensorTemperature5',
				'windingTemp1','windingTemp2','windingTemp3','windingTemp4','windingTemp5',
				'windingTempFlt1','windingTempFlt2','windingTempFlt3','windingTempFlt4','windingTempFlt5',
				'epsTau1','epsTau2','epsTau3','epsTau4','epsTau5'])

				#Add all of the data into the respecitve column
				for i in range(self.verify_set.time.size):
					writer.writerow([self.verify_set.time[i],
						self.verify_set.position[0,i],
						self.verify_set.position[1,i],
						self.verify_set.position[2,i],
						self.verify_set.position[3,i],
						self.verify_set.position[4,i],
						self.verify_set.positionCmd[0,i],
						self.verify_set.positionCmd[1,i],
						self.verify_set.positionCmd[2,i],
						self.verify_set.positionCmd[3,i],
						self.verify_set.positionCmd[4,i],
						self.verify_set.velocity[0,i],
						self.verify_set.velocity[1,i],
						self.verify_set.velocity[2,i],
						self.verify_set.velocity[3,i],
						self.verify_set.velocity[4,i],
						self.verify_set.velocityCmd[0,i],
						self.verify_set.velocityCmd[1,i],
						self.verify_set.velocityCmd[2,i],
						self.verify_set.velocityCmd[3,i],
						self.verify_set.velocityCmd[4,i],
						self.verify_set.velocityFlt[0,i],
						self.verify_set.velocityFlt[1,i],
						self.verify_set.velocityFlt[2,i],
						self.verify_set.velocityFlt[3,i],
						self.verify_set.velocityFlt[4,i],
						self.verify_set.accel[0,i],
						self.verify_set.accel[1,i],
						self.verify_set.accel[2,i],
						self.verify_set.accel[3,i],
						self.verify_set.accel[4,i],
						self.verify_set.accelCmd[0,i],
						self.verify_set.accelCmd[1,i],
						self.verify_set.accelCmd[2,i],
						self.verify_set.accelCmd[3,i],
						self.verify_set.accelCmd[4,i],
						self.verify_set.torque[0,i],
						self.verify_set.torque[1,i],
						self.verify_set.torque[2,i],
						self.verify_set.torque[3,i],
						self.verify_set.torque[4,i],
						self.verify_set.torqueCmd[0,i],
						self.verify_set.torqueCmd[1,i],
						self.verify_set.torqueCmd[2,i],
						self.verify_set.torqueCmd[3,i],
						self.verify_set.torqueCmd[4,i],
						self.verify_set.torqueID[0,i],
						self.verify_set.torqueID[1,i],
						self.verify_set.torqueID[2,i],
						self.verify_set.torqueID[3,i],
						self.verify_set.torqueID[4,i],
						self.verify_set.deflection[0,i],
						self.verify_set.deflection[1,i],
						self.verify_set.deflection[2,i],
						self.verify_set.deflection[3,i],
						self.verify_set.deflection[4,i],
						self.verify_set.deflection_vel[0,i],
						self.verify_set.deflection_vel[1,i],
						self.verify_set.deflection_vel[2,i],
						self.verify_set.deflection_vel[3,i],
						self.verify_set.deflection_vel[4,i],
						self.verify_set.motorSensorTemperature[0,i],
						self.verify_set.motorSensorTemperature[1,i],
						self.verify_set.motorSensorTemperature[2,i],
						self.verify_set.motorSensorTemperature[3,i],
						self.verify_set.motorSensorTemperature[4,i],
						self.verify_set.windingTemp[0,i],
						self.verify_set.windingTemp[1,i],
						self.verify_set.windingTemp[2,i],
						self.verify_set.windingTemp[3,i],
						self.verify_set.windingTemp[4,i],
						self.verify_set.windingTempFlt[0,i],
						self.verify_set.windingTempFlt[1,i],
						self.verify_set.windingTempFlt[2,i],
						self.verify_set.windingTempFlt[3,i],
						self.verify_set.windingTempFlt[4,i],
						self.verify_set.epsTau[0,i],
						self.verify_set.epsTau[1,i],
						self.verify_set.epsTau[2,i],
						self.verify_set.epsTau[3,i],
						self.verify_set.epsTau[4,i]])

if __name__ == '__main__':
	#Parsing inputs for plotting, trajectory generation, and saving options
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--plot", type=str,default="none",help="plots to run (default: none)")
	parser.add_argument("-t", "--traj", type=str,default="pnp",help="trajectory to run (default: PnP)")
	parser.add_argument("-s", "--save", type=str,default="no",help="save sets to csv files (default: no)")
	args = parser.parse_args()
	plotting = args.plot.lower()
	traj_type = args.traj.lower()
	saving = (args.save.lower()=="yes")

	#Creating a start position namedtuple
	position = namedtuple('Position', ['c_x', 'c_y','c_z','theta'])

	try:
		#MotorOn is a convenience variable for testing the code with/without motor
		#commands or the model learning process
		# 0. = motor off but code runs
		# 1. = motor on with model learning commands
		# 2. = motor on without model learning commands
		motorOn = 2.

		#Initialize the ros node
		rospy.init_node('model_learner', anonymous=True)
		check_reset = False
		
		cap = 3000 #number of datapoints to cap at
		gaus_noise = 1.25 #fixed gaussian noise parameter for the GP model

		#Setup control struct
		control_info = {}
		if traj_type == "circle":
			control_info['type'] = "Circle"
			position.c_x = 0.0
			position.c_y = 0.395
			position.c_z = 0.0
			position.theta = 0
			control_info['tf'] = 10.
		else:
			control_info['type'] = "MinJerk"
			position.c_x = 0.0
			position.c_y = 0.395
			position.c_z = 0.1
			position.theta = 0
			control_info['tf'] = 16.
		control_info['p_gain'] = [25.,25.,20.,10.,3.];
		control_info['v_gain'] = [0.1,0.1,0.1,0.1,0.1];
		control_info['Set'] = "train"
		control_info['closedLoop'] = True
		control_info['feedforward'] = True
		control_info['ml'] = False

		###### Training the Model ######
		ps = pubSub(bag=False,control_info=control_info)
		resetPosition(motorOn,ps,position)
		ps.reset()
		db = modelDatabase(ps)#,deflection=True)
		db.data_cap = cap
		db.setJointsToLearn(np.array([0,1,2,3,4]))
		db.controller(motorOn,control_info)
		resetPosition(motorOn,ps,position)
		db.updateSet(New=True,Set="train")
		db.downSample(Set="train")

		#Update and potentially optimize models			
		db.updateModel(optimize=True,gaus_noise=gaus_noise)
		ps.unregister()
		time.sleep(1)

		###### Verifying the Model ######
		#Update the control_info for the next run
		control_info['Set'] = "verify"
		control_info['ml'] = True
		ps = pubSub(bag=False,control_info=control_info)
		db.ps = ps
		ps.reset()
		motorOn = 1.
		db.data_cap = cap
		db.setJointsToLearn(np.array([0,1,2,3,4]))
		db.controller(motorOn,control_info)
		resetPosition(motorOn,ps,position)
		db.updateSet(New=True,Set="verify")
		ps.unregister()

		#Plotting the train and verify set before relearning occcurs
		for k in range(db.joints_ML.size):
			if plotting == "miminal" or plotting == "all": #Minimal plotting is only position
				plt.figure(10+k) #Plots in the 10's range designate position
				plt.plot(db.train_set.time,db.train_set.position[db.joints_ML[k]],linewidth=3,label='RBD',color='b')
				plt.plot(db.verify_set.time,db.verify_set.position[db.joints_ML[k]],linewidth=3,label='Task-Based GP Trial 1',color='m')
			
			if plotting == "all": #All plotting is position, velocity, and torque
				plt.figure(20+k) #Plots in the 20's range designate velocity
				plt.plot(db.train_set.time,db.train_set.velocityFlt[db.joints_ML[k]],linewidth=3,label='RBD',color='b')
				plt.plot(db.verify_set.time,db.verify_set.velocityFlt[db.joints_ML[k]],linewidth=3,label='Task-Based GP Trial 1',color='m')
		

		###### Retraining the Model ######
		control_info['Set'] = "train"
		ps = pubSub(bag=False,control_info=control_info)
		db.ps = ps
		ps.reset()
		motorOn = 1.
		db.data_cap = cap
		db.setJointsToLearn(np.array([0,1,2,3,4]))
		db.controller(motorOn,control_info)
		resetPosition(motorOn,ps,position)
		db.updateSet(New=False,Set="train")
		db.downSample(Set="train")
		db.updateModel(optimize=False,gaus_noise=gaus_noise)
		ps.unregister()

		##### Testing the Closed Loop Model ######
		control_info['Set'] = "test"
		ps = pubSub(bag=False,control_info=control_info)
		db.ps = ps
		ps.reset()
		motorOn = 1.
		db.controller(motorOn,control_info)
		resetPosition(motorOn,ps,position)
		db.updateSet(New=True,Set="test")

		if saving:
			db.saveDatabase()
			print "Data sets saved"

		ps.unregister()

		

		#### TRACKING PLOTS #####
		#Tracking plots and RMSE for each one of the learned joints
		for k in range(db.joints_ML.size):
			idx = db.joints_ML[k] #Index corresponding with the joint number
			#Minimal plotting is only position
			if plotting == "miminal" or plotting == "all":
				#Compute the position tracking RMSE for the three tasks: RBD, Task-Based GP Trial 1 (GP1),
				#Task-Based GP Trial 2 (GP2)			
				RMSE_RBD_pos = np.sqrt(np.mean(np.square(db.train_set.positionCmd[db.joints_ML[k],:final_index]-db.train_set.position[db.joints_ML[k],:final_index])))
				RMSE_GP1_pos = np.sqrt(np.mean(np.square(db.verify_set.positionCmd[db.joints_ML[k],:final_index]-db.verify_set.position[db.joints_ML[k],:final_index])))
				RMSE_GP2_pos = np.sqrt(np.mean(np.square(db.test_set.positionCmd[db.joints_ML[k]]-db.test_set.position[db.joints_ML[k]])))
				#Print to the screen the Position RMSE and the percent imporvement
				print "Joint",k+1,"Pos_RMSE_RBD: ", RMSE_RBD_pos
				print "Joint",k+1,"Pos_RMSE_GP1: ", RMSE_GP1_pos
				print "Joint",k+1,"Pos_RMSE_GP2: ", RMSE_GP2_pos
				print "Joint",k+1,"Percentage Improvement in Position GP1: ", (RMSE_RBD_pos-RMSE_GP1_pos)/RMSE_RBD_pos*100
				print "Joint",k+1,"Percentage Improvement in Position GP2: ", (RMSE_RBD_pos-RMSE_GP2_pos)/RMSE_RBD_pos*100
				
				plt.figure(10+k) #Plots in the 10's range designate position
				plt.plot(db.test_set.time,db.test_set.position[db.joints_ML[k]],linewidth=3,color="green",label='Task-Based GP Trial 2')
				plt.plot(db.test_set.time,db.test_set.positionCmd[db.joints_ML[k]],'r--',linewidth=3,label='Desired')
				plt.title('Joint Position Tracking Comparison for Task-Based GP')
				plt.legend()
				plt.ylabel('Position [rad]')
				plt.xlabel('Time [s]')
			#All plotting is position, velocity, and torque
			if plotting == "all":
				#Compute the velocity tracking RMSE for the three tasks: RBD, Task-Based GP Trial 1 (GP1),
				#Task-Based GP Trial 2 (GP2)			
				RMSE_RBD_vel = np.sqrt(np.mean(np.square(db.train_set.velocityCmd[db.joints_ML[k],:final_index]-db.train_set.velocityFlt[db.joints_ML[k],:final_index])))
				RMSE_GP1_vel = np.sqrt(np.mean(np.square(db.verify_set.velocityCmd[db.joints_ML[k],:final_index]-db.verify_set.velocityFlt[db.joints_ML[k],:final_index])))
				RMSE_GP2_vel = np.sqrt(np.mean(np.square(db.test_set.velocityCmd[db.joints_ML[k]]-db.test_set.velocityFlt[db.joints_ML[k]])))
				#Print to the screen the velocity RMSE and the percent imporvement
				print "Joint",k+1,"Vel_RMSE_RBD: ", RMSE_RBD_vel
				print "Joint",k+1,"Vel_RMSE_GP1: ", RMSE_GP1_vel
				print "Joint",k+1,"Vel_RMSE_GP2: ", RMSE_GP2_vel
				print "Joint",k+1,"Percentage Improvement in Velocity GP1: ", (RMSE_RBD_vel-RMSE_GP1_vel)/RMSE_RBD_vel*100
				print "Joint",k+1,"Percentage Improvement in Velocity GP2: ", (RMSE_RBD_vel-RMSE_GP2_vel)/RMSE_RBD_vel*100
				
				plt.figure(20+k) #Plots in the 20's range designate velocity
				plt.plot(db.test_set.time,db.test_set.velocityFlt[db.joints_ML[k]],linewidth=3,color="green",label='Task-Based GP Trial 2')
				plt.plot(db.test_set.time,db.test_set.velocityCmd[db.joints_ML[k]],'r--',linewidth=3,label='Desired')
				plt.title('Joint Velocity Tracking Comparison for Task-Based GP')
				plt.legend()
				plt.ylabel('Velocity [rad/s]')
				plt.xlabel('Time [s]')

				plt.figure(30+k) #Plots in the 30's range designate torque
				plt.plot(db.test_set.time,db.test_set.torqueCmd[db.joints_ML[k]],color="green",label='Commanded Torque (GP Trial 2)')
				plt.plot(db.test_set.time,db.test_set.torqueID[db.joints_ML[k]],color="blue", label='RBD Torque  (GP Trial 2)')
				plt.plot(db.test_set.time,db.test_set.torqueID[db.joints_ML[k]]+db.test_set.epsTau[db.joints_ML[k]],color="red",label='GP Torque (GP Trial 2)')
				plt.title('Joint Torque Tracking Comparison for Task-Based GP Trial 2')
				plt.legend()
				plt.ylabel('Torque [N-m]')
				plt.xlabel('Time [s]')

		plt.show()
	
	#If Python exception thrown, reset the arm position using position
	#control before raising the error again to be caught by the system
	except:
		#Specify the desired namedtuple position to move to
		position.c_x = 0.0
		position.c_y = 0.395
		position.c_z = 0.1
		position.theta = 0.
		#Recreating pubSub class in case the error comes befor this step
		ps = pubSub()	
		resetPosition(motorOn,ps,position)
		raise
