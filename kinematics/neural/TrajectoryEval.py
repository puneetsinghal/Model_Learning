import numpy as np
import matplotlib.pyplot as plt

def trajectoryLimits(trajFun,t0,tf):
	vel_limit = 0.95*np.array([1.5,1.5,1.5,3.43,9.6])
	vel_limit.shape = (5,1)
	vel_limit_flag = False
	dt = 0.001
	dtime = np.arange(float(t0),float(tf),dt)

	for t in dtime:
		if t == t0:
			pos,vel,accel = trajFun(t0)
			pos.shape = (5,1)
			vel.shape = (5,1)
			accel.shape = (5,1)

			if (vel > vel_limit).any() and not vel_limit_flag:
				print "Trajectory exceeds velocity limit"
				vel_limit_flag = True

		else:
			pos_v,vel_v,accel_v = trajFun(t)
			pos_v.shape = (5,1)
			vel_v.shape = (5,1)
			accel_v.shape = (5,1)

			if (vel_v > vel_limit).any() and not vel_limit_flag:
				print "Trajectory exceeds velocity limit"
				vel_limit_flag = True

			pos = np.hstack((pos,pos_v))
			vel = np.hstack((vel,vel_v))
			accel = np.hstack((accel,accel_v))

	# f_pos, axarr_pos = plt.subplots(5, sharex=True)
	f_vel, axarr_vel = plt.subplots(5, sharex=True)
	# f_accel, axarr_accel = plt.subplots(5, sharex=True)

	vel_limit_vec = np.ones((5,dtime.size))

	for i in range(5):
		vel_limit_vec[i] = vel_limit[i]*vel_limit_vec[i]
		axarr_vel[i].plot(dtime, vel[i])
		axarr_vel[i].plot(dtime,vel_limit_vec[i],'r--')
		axarr_vel[i].plot(dtime,-vel_limit_vec[i],'r--')

	plt.show()
