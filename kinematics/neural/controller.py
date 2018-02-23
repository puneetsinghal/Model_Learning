#!/usr/bin/env python

#ROS
import rospy
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import std_msgs.msg
import rosbag
import model_learning.msg

#Standard Python Libraries
import sys
import csv
import math
from datetime import datetime
import time
import argparse
from collections import namedtuple

#Third Party Software
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

import os

#Python Script
import TrajectoryGenerartor as TG

class dataStruct:
	#Class for the initialization of a dataset struct with numpy
	#array members
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

def resetPosition(motorOn,ps,position):
	# Reset the arm back to the start position using a position controller with grav comp
	#	motorOn [in]	= specifier for whether to actuate the system (convenience feature for testing)
	#	ps [in]			= publisher subscriber class for communicating to the modules
	# 	position [in]	= position named tuple designating start configuration

	tf = 4.		#Length of time to send the command
	start_time = rospy.Time.now()	#Get the current time
	time_from_start = 0.

	cmd = model_learning.msg.CommandML()
	point = JointTrajectoryPoint()

	#Perform inverse kinematics on the input position named tuple
	point.positions = TG.inverseKinematics(position.c_x,position.c_y,position.c_z,position.theta)
	point.velocities = [0.,0.,0.,0.,0.]

	#Populate the command struct
	cmd.epsTau = [0.,0.,0.,0.,0.]
	cmd.jointTrajectory.points.append(point)
	cmd.motorOn = float(motorOn)
	cmd.controlType = 0.		#controlType designated as position control

	#Send the position command until the final time is reached
	#Assumes it reaches the point in that time and avoids an actual confirmation
	while tf- time_from_start> 0:
		current_time = rospy.Time.now()
		cmd.jointTrajectory.header.stamp = current_time
		time_from_start = ((current_time-start_time).secs
						  +(current_time-start_time).nsecs/1000000000.)
		ps.traj_pub.publish(cmd)

class pubSub:
	# Publisher and subscriber class for communicating with the C++ arm_controller
	# via custom ROS messages (FeedbackML and CommandML)
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
						   #actually putting any of the data into the queue

		#Create a ROS publisher for sending command signals to the arm controller
		self.traj_pub = rospy.Publisher("ml_publisher",
										model_learning.msg.CommandML, queue_size=1)

		# Create a ROS subscriber for collecting the arm controller feedback into a queue to potentially
		# input into one of the modeldatabase datasets
		# Queue size is large since Python is substantially slower at processing the data than the arm
		# controller
		self.fbk_sub = rospy.Subscriber("armcontroller_fbk",
							model_learning.msg.FeedbackML, self.armControlCallback,
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
		#	ros_data[in]	= FeedbackML custom ROS message
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

def controller(ps,motorOn,control_info):
	#Controller for creating the command message to pass to the arm_controller
	# 	motorOn [in]	= specifier for whether to actuate the system (convenience feature for testing)
	#	control_info [in]= control structure specifying the relevant control parameters

	start_time = rospy.Time.now()
	time_from_start = 0.
	ps.final_time=control_info['tf']
	if ps.startBag:
		ps.bagging = True

	#MinJerk trajectory type specifies waypoints to move to with zero velocity and accelration
	#boundary conditions
	if control_info['type'] == "MinJerk":
			waypoints = np.array([[-0.02, -0.02, -0.02, 0.365,  0.365, 0.365,   0.0,   0.0],
								  [ 0.49,  0.49,  0.49, 0.2175,  0.2175, 0.2175, 0.395, 0.395],
								  [  0.1,   -0.02,   0.1,  0.1,   -0.02,  0.1,   0.1,   0.1],
								  [   0.,   0.0,    0.,np.pi/2,np.pi/2,  0.0,   0.0,   0.0]])
			#Specify time interval for each segment
			time_array = np.array([   2.,    2.,    2.,   2.,    2.,   2.,  2.,   2.])
			#Compute the total time needed
			tf = np.sum(time_array)
			#Use the inverseKinematics function to generate the starting joint configuration
			initial_angles= TG.inverseKinematics(0.0,0.395,0.1,0)

			#Compute the minimum jerk constants
			joint_const = TG.minJerkSetup_now(initial_angles,tf,waypoints,t_array=time_array)

	#Loop for the duration of the control interval
	while control_info['tf']-time_from_start> 0:
		deflection = ps.queue.deflection[:,-1]
		temperature = ps.queue.windingTempFlt[:,-1]
		cmd = model_learning.msg.CommandML()
		trajectory = JointTrajectory()
		point = JointTrajectoryPoint()
		trajectory.header.stamp = rospy.Time.now()
		point.time_from_start = trajectory.header.stamp-start_time
		time_from_start = (point.time_from_start.secs
						  + point.time_from_start.nsecs/1000000000.)

		#Compute the next time step for the minimum jerk trajectory
		if control_info['type'] == "MinJerk":
			pos_v,vel_v,accel_v = TG.minJerkStep_now(time_from_start,tf,waypoints,joint_const,t_array=time_array)

		#For each of the joints, populate the command trajectory
		for joint in range(0,5):
			if joint == 0 or joint==1 or joint == 2 or joint == 3 or joint == 4:
				if control_info['type'] == "Circle":
					pos_v,vel_v,accel_v = TG.generateJointTrajectory_now(time_from_start)
					pos = pos_v[joint]
					vel = vel_v[joint]
					accel = accel_v[joint]
				elif control_info['type'] == "MinJerk":
					pos = pos_v[joint]
					vel = vel_v[joint]
					accel = accel_v[joint]
			else:
				pos,vel,accel = 0.,0.,0.
			point.positions.append(pos)
			point.velocities.append(vel)
			point.accelerations.append(accel)
			 
		first_joint = 0
		eps = [0.,0.,0.,0.,0.]

		#Popuylate the command message with appropriate values
		cmd.epsTau = eps
		cmd.jointTrajectory.points.append(point)
		cmd.motorOn = float(motorOn)
		cmd.controlType = 1.		# Designates torque control
		cmd.closedLoop = control_info['closedLoop']
		cmd.feedforward = control_info['feedforward']
		cmd.pos_gain = control_info['p_gain']
		cmd.vel_gain = control_info['v_gain']
		ps.traj_pub.publish(cmd)

	if ps.startBag:
		ps.startBag = False
		ps.bagging = False
		time.sleep(0.001)
		ps.bag.close()

if __name__ == '__main__':
	#Parsing inputs for plotting, trajectory generation, and saving options
	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--traj", type=str,default="pnp")

	args = parser.parse_args()
	traj_type = args.traj.lower()

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

		#Setup control struct/dictionary
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
		controller(ps,motorOn,control_info)
		resetPosition(motorOn,ps,position)

		ps.unregister()
	
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
