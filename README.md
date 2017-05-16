# Model_Learning
This is a GP model learning package for the HEBI Robotics 5-DOF manipulator arm in the Biorobotics Lab. It mainly consists of a python script ML_Feedback.py which handles the database management and GP learning along with the arm_controller.cpp file which communicates to the modules directly as well as computes the rigid body dynamic torques.
## Dependencies
### OS
Ubuntu 16.04
### ROS
[ROS Kinetic Kame](http://wiki.ros.org/kinetic/Installation/Ubuntu)
### Python (2.7.12)
1. [GPy](https://github.com/SheffieldML/GPy)
- Install GPy with Pip
```
	sudo apt install python-pip
	pip install gpy
```
### C++ (Standard 11)
1. [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
```
sudo apt install libeigen3-dev
```
## Implementation Specifics
In this section, I cover some of the specific things which to be aware of when using this code.
### Torque Control
Except for the position reset command, the module controller is purely torque base and none of the HEBI low level loops are use. The torque feedforward command acts solely as a conversion to a PWM signal. The most important takeaway from this is that signal dropouts or not having a control loop around position will cause the arm to drift under a constant command. This is especially true when it comes to use on the lab network. Around 75% of problems I have found with the arm underperforming were due to the signal being sent over to the lab network. On the network, only ~70-90Hz can be achieved (and is very unreliable) while ~400Hz can be accomplished with a dedicated router. I would highly recommend having a dedicated router in place for testing.
### Trajectory Generation
The trajectory generator has some simple tasks (circle and infinity-loop) along with a minimum jerk waypoint generator that can be used to test the arm. However, in all three cases, the user is responsible for adjusting the time of the trajectory. The trajectory generator does not consider the physical limits of the model and allows you to put in as short a time as desired. A short timeframe can cause the trajectory to be impossible to achieve and model learning performance is degraded.
### HEBI Module Frequency
The HEBI modules are capped at a maximum frequency of 1kHz, which includes both feedback and command signals. This is something that is partially accounted for in the code with delays. However, if the control signal is sent at a higher frequency than 1kHz, the modules will begin dropping the majority of packages and the feedback frequency will drop to a very low value (~5-20Hz). Additionally, the setting of the feedback frequency of the modules in the arm_controller init function only sets the maximum frequency and is still constrained by the 1kHz limit.

