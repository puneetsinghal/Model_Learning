
#Third Party Software
import numpy as np

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
