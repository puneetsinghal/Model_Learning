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


import numpy as np
from dataStruct import dataStruct

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