/*! \mainpage Arm Controller C++ Source Code
 * The main subsections of this code are the kinematics and dynamics namespaces, 
 * which have classes for calculating kinematics and dynamics respectively, as well
 * as the arm_controller class, which is the main executable and performs both
 * communication with the HEBI modules as well as message passing to the model
 * learning script. Since the HEBI documentation is also based in doxygen notation,
 * the HEBI comments are included although this was not desired to be the case.
 */

//! Arm Controller Node for the HEBI Robotics 5-DOF Manipulator Arm
/*! Run the arm controller node that receives feedback from
	and sends commands to the HEBI modules

		@Author Ky Woodard
*/

//C++ Standard Libraries
#include <chrono>
#include <sstream>
#include <deque>
#include <cstdio>
#include <cmath>

//3rd Party Libraries
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Geometry>

//Source Files
#include <arm_dynamics.hpp>

//HEBI
#include "lookup.hpp"
#include "module.hpp"
#include "group.hpp"
#include "command.hpp"
#include "mac_address.hpp"
#include "lookup_helpers.cpp"
#include "hebi_util.h"

//ROS
#include "ros/ros.h"

//Standard ROS messages
#include "std_msgs/String.h"
#include "geometry_msgs/Pose.h"
#include "sensor_msgs/Temperature.h"
#include "sensor_msgs/JointState.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"

//Custom ROS messages
#include <model_learning/CommandML.h>
#include <model_learning/FeedbackML.h>



class armcontroller
{
private:
	long timeoutFbk_ms = 10;
	double controlType = 0;
	double maxDt = 0;
	double fbksuccess_count = 1;
	double fbkfailure_count = 1;

	model_learning::CommandML currentCmd;

	model_learning::FeedbackML currentFbk;
	sensor_msgs::JointState jointState;
	trajectory_msgs::JointTrajectoryPoint trajectoryCmd;

	std::vector<std::deque<double>> velocity_que;
	std::vector<std::deque<double>> temperature_que;

	std::vector<double> WMA_vel_single;
	std::vector<double> numerator_vel_single;
	std::vector<double> total_vel_single;
	std::vector<double> WMA_temp_single;
	std::vector<double> numerator_temp_single;
	std::vector<double> total_temp_single;

	ros::WallTime prevTime;
	std::vector<double> prevVelFlt;
	std::vector<double> prevDeflection;

	int num_vel_filt = 20;
	int num_temp_filt = 10;

	std::unique_ptr<hebi::Group> arm_group;
	bool prelim_setup;
	bool initialize;
	bool initializePosition;
	std::vector<double> initial_position;

	ros::WallTime lastCommand_time;

public:
	/*! Constructor for the arm controller, which initializes values used
	*/
	armcontroller();
	/*! Initialization function sets up the communication with the modules
	*/
	void init();
	/*! Asks for feedback from the modules and if received, updates the current
	*	controller feedback state.  It additionally calls the filtering and
	*	dynamics computation functions.
	*/
	bool updateFeedback();
	/*! Getter function for the feedback to pull the most recent feedback call
	*	@param[out] jointState_fbk ROS standard JointState message
	*	@param[out] trajectoryCmd_fbk ROS standard JointTrajectory message
	*	@param[out] armcontroller_fbk Custom message encapsilating jointState
	*				JointTrajectory and all custom inputs
	*	@return flag signifying if the Feedback was update successfully
	*/
	void getFeedbackMsg(sensor_msgs::JointState &jointState_fbk,
				trajectory_msgs::JointTrajectoryPoint &trajectoryCmd_fbk,
				model_learning::FeedbackML &armcontroller_fbk);
	/*! Generates the torque command based off the feedback and feedforward
	*	control components
	*	@param[in] positionCmd desired joint position commands [rad]
	*	@param[in] velocityCmd desired joint velocity commands [rad/s]
	*	@param[in] accelCmd desired joint acceleration commands [rad/s^s]
	*	@param[out] torque module input torques [N-m]
	*/
	void controller(const Eigen::VectorXd &positionCmd,
		   const Eigen::VectorXd &velocityCmd, const Eigen::VectorXd &accelCmd,
			std::vector<double> &torque);
	/*! Linear feedback controller on position and velocity
	*	@param[in] error_pos joint position errors [rad]
	*	@param[in] error_vel joint velocity errors [rad/s]
	*	@param[out] torque joint control torques [N-m]
	*/
	void feedbackControl(const Eigen::VectorXd &error_pos,
				const Eigen::VectorXd &error_vel,std::vector<double> &torque);
	/*! Sends position and torque commands to the modules
	*	@param[in] alpha joint position signal [rad]
	*	@param[in] torque joint torque signal [N-m]
	*	@return flag signifying if the command was sent
	*	(does not verify the module got it)
	*/
	bool sendCommand(const std::vector<double> &alpha,
					 const std::vector<double> &torque);
	/*! Sends torque commands to the modules
	*	@param[in] torque joint torque signal [N-m]
	*	@return flag signifying if the command was sent
	*	(does not verify the module got it)
	*/
	bool sendTorqueCommand(const std::vector<double> &torque);
	/*! Weighted moving average filter for velocity and temperature
	*	@param[in] velocity unfiltered velocity signal [rad/s]
	*	@param[in] motorTemp unfiltered motor temperature [C]
	*	@param[out] velocityFlt filtered velocity signal [rad/s]
	*	@param[out] motorTempFlt filtered motor temperature [C]
	*/
	void WMAFilter(const std::vector<double> &velocity,
				   const std::vector<double> &motorTemp,
				   std::vector<double> &velocityFlt,
				   std::vector<double> &motorTempFlt);
	/*! Weighted moving average filter
	*	@param[in] data_vector unfiltered signal
	*	@param[in] num number of past points to filter
	*	@param[in] vec_sum sum of all the weighting values
	*	@param[in,out] WMA weighted moving average (incrementally updated)
	*	@param[in,out] numerator sum of all the ponit multiplied by their
	*				   respective weights (incrementally updated)
	*	@param[in,out] total sum of all the points (incrementatlly updated)
	*/
	void movingAverage(const std::vector<double> &data_vector,const int &num,
					   const int &vec_sum,double &WMA,double &numerator,
					   double &total) const;
	/*! Callback function for getting the commands from the CommandML ros topic
	*	and performing the control actions
	*	@param[in] cmd custom command message from ROS topic callback
	*/
	void subscriberCallback(const model_learning::CommandML &cmd);
};

armcontroller::armcontroller()
{
	// Initializing variables to be used in this class
	// Setup Flags
	this->prelim_setup = false;
	this->initialize = false;
	this->initializePosition = false;

	// WMA Filtering vectors
	this->velocity_que = 
				std::vector<std::deque<double>>(5,std::deque<double>(5,0));
	this->temperature_que = 
				std::vector<std::deque<double>>(5,std::deque<double>(5,0));
	WMA_vel_single.resize(5);
	numerator_vel_single.resize(5);
	total_vel_single.resize(5);
	WMA_temp_single.resize(5);
	numerator_temp_single.resize(5);
	total_temp_single.resize(5);

	// Finite difference vectors
	prevDeflection.resize(5);
	prevVelFlt.resize(5);

	// Initializing all the message values
	trajectory_msgs::JointTrajectoryPoint point;
	point.positions.resize(5);
	point.velocities.resize(5);
	point.accelerations.resize(5);
	this->currentCmd.jointTrajectory.points.push_back(point);
	this->currentCmd.epsTau.resize(5);
	this->currentCmd.pos_gain.resize(5);
	this->currentCmd.vel_gain.resize(5);
	this->currentCmd.motorOn = 0.;
	this->currentCmd.closedLoop = false;
	this->currentCmd.feedforward = false;
	this->trajectoryCmd.positions.resize(5);
	this->trajectoryCmd.velocities.resize(5);
	this->trajectoryCmd.accelerations.resize(5);
	this->jointState.position.resize(5);
	this->jointState.velocity.resize(5);
	this->jointState.effort.resize(5);
	this->currentFbk.trajectoryCmd.positions.resize(5);
	this->currentFbk.trajectoryCmd.velocities.resize(5);
	this->currentFbk.trajectoryCmd.accelerations.resize(5);
	this->currentFbk.jointState.position.resize(5);
	this->currentFbk.jointState.velocity.resize(5);
	this->currentFbk.jointState.effort.resize(5);
	this->currentFbk.deflections.resize(5);
	this->currentFbk.velocityFlt.resize(5);
	this->currentFbk.deflection_vel.resize(5);
	this->currentFbk.motorSensorTemperature.resize(5);
	this->currentFbk.windingTemp.resize(5);
	this->currentFbk.windingTempFlt.resize(5);
	this->currentFbk.torqueCmd.resize(5);
	this->currentFbk.torqueID.resize(5);
	this->currentFbk.accel.resize(5);
	this->currentFbk.epsTau.resize(5);

	// Initializing the command start time
	this->lastCommand_time = ros::WallTime::now();
}

void armcontroller::init()
{
	long timeout_ms = 5000; // Give the modules plenty of time to appear.
	float freq_hz = 1000;	// Feedback frequency of the modules

	hebi::Lookup lookup;
	std::vector<hebi::MacAddress> macs;

	// Lab Research Arm's Mac Addresses
	std::vector<std::string> modules = {"D8:80:39:E8:B3:3C",
										"D8:80:39:E9:06:36",
										"D8:80:39:9D:29:4E",
										"D8:80:39:9D:3F:9D",
										"D8:80:39:9D:04:78"};

	// Build the vector of mac addresses and check for correct
	// hex strings
	for(int i=0;i<5;i++)
	{
		hebi::MacAddress mac;
		if(hebi::MacAddress::isHexStringValid(modules[i]))
		{
			mac.setToHexString(modules[i]);
			macs.emplace_back(mac);
		}
		else
		{
			perror("Module Hex String is Invalid");
		}
	}

	// Set up the arm_group from the mac addresses
	arm_group = lookup.getGroupFromMacs(macs,timeout_ms);
	hebi_sleep_ms(100);
	arm_group->setFeedbackFrequencyHz(freq_hz);
	hebi_sleep_ms(100);
	if(!arm_group)
	{
		perror("Failed to get module group");
	}
}

bool armcontroller::updateFeedback()
{
	// Define the HEBI group feedback class
	hebi::GroupFeedback groupFeedback(this->arm_group->size());
	bool fbkSuccess = true;

	// Request feedback from the modules
	// Keep the timout ms at 5. This is a tradeoff between the delay being too
	// long and blocking the sequence or being too short and dropping too many
	// packets because they are not received quickly enough
	if(!arm_group->requestFeedback(&groupFeedback,5)){fbkSuccess = false;}

	// Update all of the feedback for the class with module feedback
	if(fbkSuccess)
	{
		ros::WallTime time_stamp = ros::WallTime::now();
		this->currentFbk.header.stamp.sec = time_stamp.sec;
		this->currentFbk.header.stamp.nsec = time_stamp.nsec;
		this->jointState.header.stamp.sec = time_stamp.sec;
		this->jointState.header.stamp.nsec = time_stamp.nsec;
		this->jointState.name = {std::string("Joint1"),std::string("Joint2"),
								 std::string("Joint3"),std::string("Joint4")};

		std::vector<double> velocityUpdate(5);
		std::vector<double> windingTempUpdate(5);

		for(int i=0;i<5;i++)
		{
			this->jointState.position[i] = 
								  groupFeedback[i].actuator().position().get();
			this->jointState.velocity[i] = 
								  groupFeedback[i].actuator().velocity().get();
			this->jointState.effort[i] = 
									groupFeedback[i].actuator().torque().get();

			this->currentFbk.jointState.position[i] = 
								  groupFeedback[i].actuator().position().get();
			this->currentFbk.jointState.velocity[i] = 
								  groupFeedback[i].actuator().velocity().get();
			this->currentFbk.jointState.effort[i] = 
									groupFeedback[i].actuator().torque().get();

			if(!this->initialize)
			{
				this->initial_position = this->jointState.position;
				ros::WallTime prevTime = ros::WallTime::now();
			}

			this->trajectoryCmd.positions[i] = 
					   this->currentCmd.jointTrajectory.points[0].positions[i];
			this->trajectoryCmd.velocities[i] = 
					  this->currentCmd.jointTrajectory.points[0].velocities[i];
			this->trajectoryCmd.accelerations[i] = 
				   this->currentCmd.jointTrajectory.points[0].accelerations[i];

			this->currentFbk.trajectoryCmd.positions[i] = 
					   this->currentCmd.jointTrajectory.points[0].positions[i];
			this->currentFbk.trajectoryCmd.velocities[i] = 
					  this->currentCmd.jointTrajectory.points[0].velocities[i];
			this->currentFbk.trajectoryCmd.accelerations[i] = 
				   this->currentCmd.jointTrajectory.points[0].accelerations[i];

			this->currentFbk.epsTau[i] = this->currentCmd.epsTau[i];

			this->currentFbk.torqueCmd[i] = 
							 groupFeedback[i].actuator().torqueCommand().get();
			this->currentFbk.deflections[i] = 
								groupFeedback[i].actuator().deflection().get();
			this->currentFbk.windingTemp[i] = 
				   groupFeedback[i].actuator().motorWindingTemperature().get();
			this->currentFbk.motorSensorTemperature[i] = 
					groupFeedback[i].actuator().motorSensorTemperature().get();

			velocityUpdate[i] = this->jointState.velocity[i];
			windingTempUpdate[i] = this->currentFbk.windingTemp[i];
		}

		// On initialization, display that the modules are initialized
		if(!this->initialize)
		{
			std::cout << "Modules Fully Initialized" << std::endl;
			this->initialize = true;
		}
		
		// Filter teh velocity and winding temperature using a weighted moving 
		// average filter
		std::vector<double> velocityFiltered(5);
		std::vector<double> windingTempFiltered(5);
		WMAFilter(velocityUpdate, windingTempUpdate, velocityFiltered,
				  windingTempFiltered);

		Eigen::VectorXd positionMeas(5);
		Eigen::VectorXd velocityMeas(5);
		Eigen::VectorXd accelMeas(5);

		// Setting up initial parameters for the accelration and deflection 
		// velocity low pass filters		
		double fc = 1;
		double dt = (time_stamp-this->prevTime).toSec();
		double alpha_accel = 1/(1+dt*2*M_PI*fc);

		fc = 1;;
		double alpha_def = 1/(1+dt*2*M_PI*fc);

		for(int i=0;i<5;i++)
		{
			 positionMeas[i] = this->currentFbk.jointState.position[i];
			 velocityMeas[i] = this->currentFbk.jointState.velocity[i];

			 // Low pass filter for the deflection velocity
			 this->currentFbk.deflection_vel[i] = 
			 				alpha_def*this->currentFbk.deflection_vel[i]
			 				+(1-alpha_def)*(this->currentFbk.deflections[i]
			 				-prevDeflection[i])/dt;

			 // Low pass filter for the joint acceleration
			 this->currentFbk.accel[i] = alpha_accel*this->currentFbk.accel[i]
			 							 +(1-alpha_accel)*(velocityFiltered[i]
			 							 -prevVelFlt[i])/dt;

			 accelMeas[i] = this->currentFbk.accel[i];
			 prevDeflection[i] = this->currentFbk.deflections[i];
			 prevVelFlt[i] = velocityFiltered[i];
		}
		// Compute the inverse dynamics torque based off the actual 
		// configuration values
		std::vector<double> torqueID(5);
		dynamics::inverseDynamics(positionMeas,velocityMeas,accelMeas,
								  torqueID);
		
		for(int i=0;i<5;i++)
		{
			this->currentFbk.velocityFlt[i] = velocityFiltered[i];
			this->currentFbk.windingTempFlt[i] = windingTempFiltered[i];
			this->currentFbk.torqueID[i] = torqueID[i];
		}
		this->prevTime = time_stamp;
	}
	return fbkSuccess;
}

void armcontroller::getFeedbackMsg(sensor_msgs::JointState &jointState_fbk,
								   trajectory_msgs::JointTrajectoryPoint& 
								   						trajectoryCmd_fbk,
								   model_learning::FeedbackML& 
								   						armcontroller_fbk)
{
	jointState_fbk = this->jointState;
	trajectoryCmd_fbk = this->trajectoryCmd;
	armcontroller_fbk = this->currentFbk;
}

void armcontroller::controller(const Eigen::VectorXd &positionCmd,
							   const Eigen::VectorXd &velocityCmd,
							   const Eigen::VectorXd &accelCmd,
							   std::vector<double> &torque)
{
	Eigen::VectorXd positionFB(5);
	Eigen::VectorXd velocityFB(5);
	Eigen::VectorXd error_pos(5);
	Eigen::VectorXd error_vel(5);
	std::vector<double> torqueID(5);
	std::vector<double> torqueFB(5);
	std::vector<double> posFbk(5);
	std::vector<double> alpha(5);
	bool closedLoop = this->currentCmd.closedLoop;
	bool feedforward = this->currentCmd.feedforward;


	positionFB << this->currentFbk.jointState.position[0],
				  this->currentFbk.jointState.position[1],
				  this->currentFbk.jointState.position[2],
				  this->currentFbk.jointState.position[3],
				  this->currentFbk.jointState.position[4];
	velocityFB << this->currentFbk.velocityFlt[0],
				  this->currentFbk.velocityFlt[1],
				  this->currentFbk.velocityFlt[2],
				  this->currentFbk.velocityFlt[3],
				  this->currentFbk.velocityFlt[4];

	// Calculating the position and velocity error
	error_pos = positionCmd-positionFB;
	error_vel = velocityCmd-velocityFB;

	// Creating a position vector for use in the gravity comp function
	for(int i=0;i<5;i++)
	{
		posFbk[i] = positionFB[i];
	}

	// Control types: 0->gravity compensation feedforward control
	//				  1->inverse dynamics feedforward control
	if(this->controlType == 0)
	{
		for(uint i=0;i<5;i++)
		{
			alpha[i] = positionCmd[i];
		}
		dynamics::gravityComp(alpha,torque);
		torque[4]=0;
		this->initializePosition = false;
	}
	else if(this->controlType == 1)
	{
		if(!this->initializePosition)
		{
			this->initial_position = this->jointState.position;
			this->initializePosition = true;
		}

		// Feedforward specifies inverse dynamics only (model learning is 
		// captured separately)
		if(feedforward)
		{
			dynamics::inverseDynamics(positionCmd,velocityCmd,accelCmd,
									  torqueID);
		}
		else
		{
			dynamics::gravityComp(posFbk,torqueID);
		}
		
		if (closedLoop) // Closed loop specities if a control effort is used
		{
			feedbackControl(error_pos,error_vel,torqueFB);
		}
		else
		{
			torqueFB = {0,0,0,0,0};
		}

		// Combinging the feedback and feedforward torque commands
		for(uint i=0;i<5;i++)
		{
			torque[i] = torqueID[i] + torqueFB[i];

			this->currentCmd.jointTrajectory.points[0].positions[i] =
																positionCmd[i];
			this->currentCmd.jointTrajectory.points[0].velocities[i] =
																velocityCmd[i];
			this->currentCmd.jointTrajectory.points[0].accelerations[i] =
																   accelCmd[i];
		}
	}
}

void armcontroller::feedbackControl(const Eigen::VectorXd &error_pos,
									const Eigen::VectorXd &error_vel,
									std::vector<double> &torque)
{
	Eigen::MatrixXd Kp(5,5);
	Eigen::MatrixXd Kv(5,5);
	Eigen::VectorXd torqueFB(5);
	Eigen::VectorXd vec_p(5);
	Eigen::VectorXd vec_v(5);

	// Setting the position and velocity gains from the command inputs
	// (allows for quick changes to gains in the Python script)
	for(int i=0;i<5;i++)
	{
		vec_p[i] = this->currentCmd.pos_gain[i];
		vec_v[i] = this->currentCmd.vel_gain[i];
	}

	// Creating a diagonal matrix for the gain values
	Kp << vec_p[0],0,0,0,0,
		  0,vec_p[1],0,0,0,
		  0,0,vec_p[2],0,0,
		  0,0,0,vec_p[3],0,
		  0,0,0,0,vec_p[4];

	Kv << vec_v[0],0,0,0,0,
		  0,vec_v[1],0,0,0,
		  0,0,vec_v[2],0,0,
		  0,0,0,vec_v[3],0,
		  0,0,0,0,vec_v[4];

	// Feedback control effort compuation
	torqueFB = Kp*error_pos + Kv*error_vel;
	for(uint i=0;i<5;i++)
	{
		torque[i] = torqueFB[i];
	}
}

bool armcontroller::sendCommand(const std::vector<double> &alpha,
								const std::vector<double> &torque)
{
	//Create a HEBI command class
	hebi::GroupCommand command(this->arm_group->size());
	bool bSuccess = true;
	
	// Populate the command class
	for(uint j = 0; j < 5; j++)
	{
		command[j].actuator().position().set(alpha[j]);
		command[j].actuator().torque().set(torque[j]);
	}

	// Send command to the modules
	if(!arm_group->sendCommand(command)){bSuccess = false;}
	if(bSuccess == false){
		// printf("Did not receive acknowledgement!\n");
	}else{
		// printf("Got acknowledgement.\n");
	}
	return bSuccess;
}

bool armcontroller::sendTorqueCommand(const std::vector<double> &torque)
{
	//Create a HEBI command class
	hebi::GroupCommand command(this->arm_group->size());
	bool bSuccess = true;

	// Populate the command class
	for(uint j = 0; j < 5; j++)
	{
		command[j].actuator().torque().set(torque[j]);
	}

	// Send command to the modules
	if(!arm_group->sendCommand(command)){bSuccess = false;}
	if(bSuccess == false){
		// printf("Did not receive acknowledgement!\n");
	}else{
		// printf("Got acknowledgement.\n");
	}
	return bSuccess;
}

void armcontroller::WMAFilter(const std::vector<double> &velocity,
							  const std::vector<double> &motorTemp,
							  std::vector<double> &velocityFlt,
							  std::vector<double> &motorTempFlt)
{
	// Add a new point to the velocity que and remove extras if it exceeds the
	// velocity filter max number
	for(int i=0;i<this->velocity_que.size();i++)
	{
		this->velocity_que[i].push_back(velocity[i]);
	
		if(this->velocity_que[i].size()>(this->num_vel_filt+1))
		{
			this->velocity_que[i].pop_front();
		}
	}
	// Compute the sum of all of weighting values
	int vel_sum;
	if(this->velocity_que[0].size()==this->num_vel_filt+1)
	{
		vel_sum = (this->velocity_que[0].size()
				 *(this->velocity_que[0].size()-1))/2;
	}
	else
	{
		vel_sum = (this->velocity_que[0].size()
				 *(this->velocity_que[0].size()+1))/2;
	}
	// Applying the weighted moving average on the velocity signal
	for(int i=0;i<this->velocity_que.size();i++)
	{
		movingAverage({this->velocity_que[i].begin(),
				   this->velocity_que[i].end()},this->num_vel_filt,vel_sum,
				   this->WMA_vel_single[i],this->numerator_vel_single[i],
				   this->total_vel_single[i]);
	}
	// Add a new point to the temperature que and remove extras if it exceeds
	// the temperature filter max number
	for(int i=0;i<this->temperature_que.size();i++)
	{
		this->temperature_que[i].push_back(motorTemp[i]);

		if(this->temperature_que[i].size()>(this->num_temp_filt+1))
		{
			this->temperature_que[i].pop_front();
		}
	}
	// Compute the sum of all of weighting values
	int temp_sum;
	if(this->temperature_que[0].size()==this->num_temp_filt+1)
	{
		temp_sum = (this->temperature_que[0].size()
				  *(this->temperature_que[0].size()-1))/2;
	}
	else
	{
		temp_sum = (this->temperature_que[0].size()
				  *(this->temperature_que[0].size()+1))/2;
	}

	// Applying the weighted moving average on the temperature signal
	for(int i=0; i<this->temperature_que.size();i++)
	{
		movingAverage({this->temperature_que[i].begin(),
					   this->temperature_que[i].end()},this->num_temp_filt,
					   temp_sum,this->WMA_temp_single[i],
					   this->numerator_temp_single[i],
					   this->total_temp_single[i]);
	}
	for(int i=0; i<5; i++)
	{
			velocityFlt[i] = this->WMA_vel_single[i];
			motorTempFlt[i] = this->WMA_temp_single[i];
	}
};

void armcontroller::movingAverage(const std::vector<double> &data_vector,
								  const int &num,const int &vec_sum,
								  double &WMA,double &numerator,
								  double &total) const
{
	//Incrementally computing the weighted moving average of an input vector
	if(((int)data_vector.size())<(num+1))
	{
		numerator = numerator + ((double)data_vector.size())*data_vector[data_vector.size()-1];
		total = total + data_vector[data_vector.size()-1];
		WMA = numerator/(double)vec_sum;
	}
	else
	{
		numerator = numerator + ((double)num)*data_vector[num]-total;
		total = total + data_vector[num] - data_vector[0];
		WMA = numerator/(double)vec_sum;
	}
};

void armcontroller::subscriberCallback(const model_learning::CommandML &cmd)
{
	ros::WallTime start = ros::WallTime::now();
	std::vector<double> torque(5);
	std::vector<double> alpha(5);
	Eigen::VectorXd positionCmd(5);
	Eigen::VectorXd velocityCmd(5);
	Eigen::VectorXd accelCmd(5);

	//Collecting command message data
	trajectory_msgs::JointTrajectory jointTrajectory = cmd.jointTrajectory;
	double motorOn = cmd.motorOn;
	double cType = cmd.controlType;
	std::vector<double> epsTau = cmd.epsTau;
	this->currentCmd.closedLoop = cmd.closedLoop;
	this->currentCmd.feedforward = cmd.feedforward;
	this->currentCmd.pos_gain = cmd.pos_gain;
	this->currentCmd.vel_gain = cmd.vel_gain;
	this->controlType = cType;

	// Control types: 0->gravity compensation feedforward control
	//				  1->inverse dynamics feedforward control
	for(uint j=0;j<5;j++)
	{
		if(cType == 0.)
		{
			positionCmd[j] = jointTrajectory.points[0].positions[j];
			alpha[j] = positionCmd[j];
		}
		else if(cType == 1.)
		{
			positionCmd[j] = jointTrajectory.points[0].positions[j];
			velocityCmd[j] = jointTrajectory.points[0].velocities[j];
			accelCmd[j] = jointTrajectory.points[0].accelerations[j];
		}
	}
	// Calling the torque controlller
	controller(positionCmd,velocityCmd,accelCmd,torque);

	//Adding in the torque error prediction if available
	for(uint j=0;j<5;j++)
	{
		 this->currentCmd.epsTau[j] = epsTau[j];
		 if (motorOn == 2.)
		 {
			 torque[j] = torque[j];
		 }
		 else
		 {
			 torque[j] = torque[j]+epsTau[j];
		 }
	}
	// Sending pure torque command
	if(motorOn && cType == 1.)
	{
		sendTorqueCommand(torque);

	}
	// Sending position and torque command
	else if(motorOn && cType == 0.)
	{
		sendCommand(alpha,torque);
	}
	
}

int main(int argc, char* argv[])
{
	// Intializing the arm controller class
	armcontroller ac;
	ac.init();

	// Setting up ROS and the publishers and subscribers
	ros::init(argc, argv, "arm_1_traj_node");
	ros::NodeHandle nh;
	ros::Publisher jointState_pub = nh.advertise<sensor_msgs::JointState>(
														"jointState_fbk",10);
	ros::Publisher trajectory_pub = 
					  nh.advertise<trajectory_msgs::JointTrajectoryPoint>(
													 "trajectoryCmd_fbk",10);
	ros::Publisher armcontroller_pub = nh.advertise<model_learning::FeedbackML>
													("armcontroller_fbk",10);
	ros::Subscriber armcontroller_sub = nh.subscribe("ml_publisher",10,
									 &armcontroller::subscriberCallback, &ac);

	// Controlling the loop rate of the controller
	ros::Rate loop_rate(100);
	std::cout << "enter ros spin" << std::endl;

	while(ros::ok()) //Checking if ros is ok slows down the loop to ~87 Hz
	{
		ros::WallTime start = ros::WallTime::now();
		sensor_msgs::JointState jointState_fbk;
		trajectory_msgs::JointTrajectoryPoint trajectoryCmd_fbk;
		model_learning::FeedbackML armcontroller_fbk;

		// Update the feedback and publish this back to the model learning
		// script
		if(ac.updateFeedback())
		{
			ac.getFeedbackMsg(jointState_fbk,trajectoryCmd_fbk,
							  armcontroller_fbk);
			jointState_pub.publish(jointState_fbk);
			trajectory_pub.publish(trajectoryCmd_fbk);
			armcontroller_pub.publish(armcontroller_fbk);
		}

		// Call all the subscriber callback functions for pending messages
		ros::spinOnce(); 
		loop_rate.sleep();  // Necessary to control the
	}
 	return 0;

}


