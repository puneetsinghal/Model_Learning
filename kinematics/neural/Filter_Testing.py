#!/usr/bin/env python
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import hebi_cpp.msg
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy.fftpack import fft

class dataStruct:
	def __init__(self):
		self.time = np.empty(shape=(1,))
		self.position = np.empty(shape=(5,1))
		self.positionCmd = np.empty(shape=(5,1))
		self.velocity = np.empty(shape=(5,1))
		self.velocityCmd = np.empty(shape=(5,1))
		self.velocityFlt = np.empty(shape=(5,1))
		self.accelCmd = np.empty(shape=(5,1))
		self.torque = np.empty(shape=(5,1))
		self.torqueCmd = np.empty(shape=(5,1))
		self.torqueID = np.empty(shape=(5,1))
		self.deflection = np.empty(shape=(5,1))
		self.windingTemp = np.empty(shape=(5,1))
		self.windingTempFlt = np.empty(shape=(5,1))

class pubSub:
	queue = dataStruct()
	restart_joint = True
	restart_traj = True
	restart_arm = True
	count = 1
	minimum = 0
	initial_time = 0.
	prev_time = 0.
	final_time = 999.

	def __init__(self):
		self.traj_pub = rospy.Publisher("ml_publisher",hebi_cpp.msg.CommandML,queue_size=1);
		self.fbk_sub = rospy.Subscriber("armcontroller_fbk",hebi_cpp.msg.FeedbackML,self.armControlCallback,queue_size=5000)

	def armControlCallback(self,ros_data):
		if self.restart_arm:
			self.addToQueue(ros_data,New=True,fbk_type="armControl")
			self.restart_arm = False
		else:
			self.addToQueue(ros_data,New=False,fbk_type="armControl")
			self.count += 1
		self.minimum = self.count
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

				self.queue.accel = np.array(np.matrix((fbk.accel)).T)
				self.queue.deflection = np.array(np.matrix((fbk.deflections)).T)
				self.queue.velocityFlt = np.array(np.matrix((fbk.velocityFlt)).T)
				self.queue.windingTemp = np.array(np.matrix((fbk.windingTemp)).T)
				self.queue.windingTempFlt = np.array(np.matrix((fbk.windingTempFlt)).T)
				self.queue.torqueCmd = np.array(np.matrix((fbk.torqueCmd)).T)
				self.queue.torqueID = np.array(np.matrix((fbk.torqueID)).T)
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

				self.queue.accel = np.hstack((self.queue.accel,np.array(np.matrix((fbk.accel)).T)))
				self.queue.deflection = np.hstack((self.queue.deflection,np.array(np.matrix((fbk.deflections)).T)))
				self.queue.velocityFlt = np.hstack((self.queue.velocityFlt,np.array(np.matrix((fbk.velocityFlt)).T)))
				self.queue.windingTemp = np.hstack((self.queue.windingTemp,np.array(np.matrix((fbk.windingTemp)).T)))
				self.queue.windingTempFlt = np.hstack((self.queue.windingTempFlt,np.array(np.matrix((fbk.windingTempFlt)).T)))
				self.queue.torqueCmd = np.hstack((self.queue.torqueCmd,np.array(np.matrix((fbk.torqueCmd)).T)))
				self.queue.torqueID = np.hstack((self.queue.torqueID,np.array(np.matrix((fbk.torqueID)).T)))
		else:
			raise ValueError("Feedback type for addToQueue is not recognized")
	def unregister(self):
		# self.state_sub.unregister()
		# self.cmd_sub.unregister()
		self.fbk_sub.unregister()
		self.fbk_sub = None
	def reregister(self):
		# self.state_sub = rospy.Subscriber("jointState_fbk",JointState,self.jointStateCallback,queue_size=1000)
		# self.cmd_sub = rospy.Subscriber("trajectoryCmd_fbk",JointTrajectoryPoint,self.trajectoryCmdCallback,queue_size=1000)
		self.fbk_sub = rospy.Subscriber("armcontroller_fbk",hebi_cpp.msg.FeedbackML,self.armControlCallback,queue_size=1000)
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

if __name__ == '__main__':
	rospy.init_node('filter_testing', anonymous=True)

	ps = pubSub()
	
	time.sleep(10)

	t = ps.queue.time
	velocity = ps.queue.velocity[0]
	velocityFlt = ps.queue.velocityFlt[0]
	velocityFltLP = np.zeros(t.size)
	accel = ps.queue.accel[0]
	accelLP = np.zeros(t.size)

	fc = 10.

	x = 0;
	for i in range(0,t.size):
		if i == 0:
			dt = 0;
		else:
			dt = t[i]-t[i-1]
		alpha = 1/(1+dt*2*np.pi*fc)
		x = alpha*x + (1-alpha)*velocity[i];
		velocityFltLP[i] = x;

	fc = 10.
	x = 0;
	for i in range(0,t.size):
		if i == 0:
			dt = 0;
		else:
			dt = t[i]-t[i-1]
		alpha = 1/(1+dt*2*np.pi*fc)
		x = alpha*x + (1-alpha)*accel[i];
		accelLP[i] = x;

	min_size = min([t.size,velocity.size,velocityFlt.size,accel.size])

	t = t[:min_size]
	velocity = velocity[:min_size]
	velocityFltLP = velocityFltLP[:min_size]
	accel = accel[:min_size]

	# xf = np.linspace(0,1000,velocity.size/2)

	# velF = fft(velocity)
	# velFltF = fft(velocityFlt)
	# print xf.shape
	# print velocity.shape

	# plt.plot(xf,2.0/velocity.size*(np.abs(velF[0:velocity.size//2])))
	plt.plot(t,velocity,label='Unfiltered')
	plt.plot(t,velocityFlt,label='WMA Filter')
	plt.plot(t,velocityFltLP, label='LP Filter (%d Hz)' % fc)
	plt.plot(t,accelLP,label='Accel')
	plt.legend()
	plt.grid()
	plt.show()

	ps.queue
	
	ps.unregister()
	
