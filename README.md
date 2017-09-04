# Model_Learning
This is a GP model learning package for the HEBI Robotics 5-DOF manipulator arm in the Biorobotics Lab. It mainly consists of a python script ML_Feedback.py which handles the database management and GP learning along with the arm_controller.cpp file which communicates to the modules directly as well as computes the rigid body dynamic torques.
## Dependencies
### OS/ROS
Ubuntu 16.04 with [ROS Kinetic Kame](http://wiki.ros.org/kinetic/Installation/Ubuntu)

Ubuntu 14.04 with [ROS Indigo Igloo](http://wiki.ros.org/indigo/Installation/Ubuntu)

### Python (2.7.6 or greater)
1. [GPy](https://github.com/SheffieldML/GPy)(1.6.1 or greater)
- Easiest Install with Pip
```
	sudo apt install python-pip
	pip install gpy
```
2. numpy (1.8.2 or greater)
3. Tkinter (8.6 or greater)
- Easiest Install with apt-get
```
	sudo apt-get install python-tk
```
4. matplotlib (1.3.1 or greater)
5. scipy (0.19.0 or greater

### C++ (Standard 11)
1. [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page) (3.2.0 or greater)
- Easiest Install with apt-get
```
	sudo apt install libeigen3-dev
```
### Matlab
1. Ardiuno Support Package (for the Hebi Arm Demos)
- https://www.mathworks.com/hardware-support/arduino-matlab.html

### OpenCV (3.2.0 or greater)

## Running the Code (ROS)
After the catkin package has been built and the setup.bash file sourced, the first step is to start a terminal and run:
```
	roscore
```

There are two executables to run in ROS in order to implement the code. The first is the arm controller in C++ that handles all the communication to the HEBI modules. This executable can be launched and run and launched in the background without closing out until all the tests are finisthed:
```
	rosrun model_learning arm_controller
```
IMPORTANT: If "Modules Fully Initialized" does not show up, the modules did not connect properly and this executable needs to be rerun. There are extreme cases where it may take >5 tries to get this to show up. The issue is only some of the modules are connected and feedback messages will all fail. The tag "Modules Fully Initialized" lets the user know that feedback is being received properly.

The second half of the code is a Python script, which does the actual learning as well as passes commands to the arm_controller. This script can be run with the following command:
```
	rosrun model_learning ML_Feedback.py
```
Python Script ML_Feedback.py Additional Parameters:
* plot (default="none","minimal","all") [shorthand -p]
	* none 	= no plots
	* minimal 	= joint position tracking plots
	* all		= joint position, velocity, and torque tracking plots
* traj (default="pnp","circle") [shorthand -t]
	* pnp		= follows a minimum jerk pick and place trajectory
	* circle	= follows a planar circular trajectory
* save (default="no","yes") [shorthand -s]
	* no		= do not save
	* yes		= saves the sets as csv files

Example Call: rosrun model_learning ML_Feedback.py --plot all -t circle -s yes

## Implementation Specifics
In this section, I cover some of the specific things which to be aware of when using this code.
### Torque Control
Except for the position reset command, the module controller is purely torque base and none of the HEBI low level loops are use. The torque feedforward command acts solely as a conversion to a PWM signal. The most important takeaway from this is that signal dropouts or not having a control loop around position will cause the arm to drift under a constant command. This is especially true when it comes to use on the lab network. Around 75% of problems I have found with the arm underperforming were due to the signal being sent over to the lab network. On the network, only ~70-90Hz can be achieved (and is very unreliable) while ~400Hz can be accomplished with a dedicated router. I would highly recommend having a dedicated router in place for testing.
### Python Error Catch
If you modify the Python main section, or any of the code really, and it causes an error, the entire main section is encapasalated in a try/catch statement so it should exit gracefully by just returning to the home position. This does no account for module feedback failure however.
### Trajectory Generation
The trajectory generator has some simple tasks (circle and infinity-loop) along with a minimum jerk waypoint generator that can be used to test the arm. However, in all three cases, the user is responsible for adjusting the time of the trajectory. The trajectory generator does not consider the physical limits of the model and allows you to put in as short a time as desired. A short timeframe can cause the trajectory to be impossible to achieve and model learning performance is degraded.
### HEBI Module Frequency
The HEBI modules are capped at a maximum frequency of 1kHz, which includes both feedback and command signals. This is something that is partially accounted for in the code with delays. However, if the control signal is sent at a higher frequency than 1kHz, the modules will begin dropping the majority of packages and the feedback frequency will drop to a very low value (~5-20Hz). Additionally, the setting of the feedback frequency of the modules in the arm_controller init function only sets the maximum frequency and is still constrained by the 1kHz limit.

## Running the Code (Hebi Arm Demos)
### Arm_Teach_Repeat_Demo.m
Teach repeat demo for the Hebi Arm. Follow the instructions on the screen.

### GrabGreenBall_v2.m
Uses the overhead camera to localize a handheld green ball and then, after 'q' is pressed, picks up ball and places the ball into a cup taped on the cart. Then enters teleoperation mode. Press ten (10) on the joystick to exit localization and relocalizes the ball.

### HEBI_Arm_Joystick_control_Demo_V2.m
Joytick teleoperation with gripper. Press 2 to turn on gripper and 3 to turn off gripper.

### HEBI_Arm_Joystick_control_Compliance_Demo.m
Joytick teleoperation without gripper. Arm will stop if contacted in the z direction.
