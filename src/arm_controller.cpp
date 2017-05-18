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


/*! Calculate the necessary torque required to cancel
						 gravitational forces on the joints
*/
class armcontroller
{
public:
	long timeoutFbk_ms = 10;
	double controlType = 0;
	double maxDt = 0;
	double fbksuccess_count = 1;
	double fbkfailure_count = 1;

	model_learning::CommandML currentCmd;

	model_learning::FeedbackML currentFbk;
	sensor_msgs::JointState jointState;
	trajectory_msgs::JointTrajectoryPoint trajectoryCmd;

	std::deque<double> single_vel_que1;
	std::deque<double> single_vel_que2;
	std::deque<double> single_vel_que3;
	std::deque<double> single_vel_que4;
	std::deque<double> single_vel_que5;

	std::deque<double> single_temp_que1;
	std::deque<double> single_temp_que2;
	std::deque<double> single_temp_que3;
	std::deque<double> single_temp_que4;
	std::deque<double> single_temp_que5;

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
	double last_alpha4;
	bool prelim_setup;
	bool initialize;
	bool initializePosition;
	std::vector<double> initial_position;

	ros::WallTime lastCommand_time;

	armcontroller();
	void init();
	bool updateFeedback();
	void getFeedbackMsg(sensor_msgs::JointState&,trajectory_msgs::JointTrajectoryPoint&,
											model_learning::FeedbackML&);
	void controller(const Eigen::VectorXd &,const Eigen::VectorXd &,
						 const Eigen::VectorXd &, std::vector<double> &);
	void feedbackControl(const Eigen::VectorXd &,const Eigen::VectorXd &,
								std::vector<double> &);
	bool sendCommand(const std::vector<double> &, const std::vector<double> &);
	bool sendTorqueCommand(const std::vector<double> &);
	void subscriberCallback(const model_learning::CommandML&);
	void WMAFilter(const std::vector<double> &, const std::vector<double> &,
												 std::vector<double> &, std::vector<double> &);
	void movingAverage(const std::vector<double> &,const int &,const int &,
									double &,double &,double &) const;
};

armcontroller::armcontroller()
{

	last_alpha4 = -200.0;
	this->prelim_setup = false;
	this->initialize = false;
	this->initializePosition = false;

	WMA_vel_single = {0,0,0,0,0};
	numerator_vel_single = {0,0,0,0,0};
	total_vel_single = {0,0,0,0,0};
	WMA_temp_single = {0,0,0,0,0};
	numerator_temp_single = {0,0,0,0,0};
	total_temp_single = {0,0,0,0,0};
	prevDeflection = {0,0,0,0,0};
	prevVelFlt = {0,0,0,0,0};

	trajectory_msgs::JointTrajectoryPoint point;
	point.positions = {0,0,0,0,0};
	point.velocities = {0,0,0,0,0};
	point.accelerations = {0,0,0,0,0};
	this->currentCmd.jointTrajectory.points.push_back(point);
	this->currentCmd.epsTau = {0,0,0,0,0};
	this->currentCmd.pos_gain = {0,0,0,0,0};
	this->currentCmd.vel_gain = {0,0,0,0,0};
	this->currentCmd.motorOn = 0.;
	this->currentCmd.closedLoop = false;
	this->currentCmd.feedforward = false;

	this->trajectoryCmd.positions = {0,0,0,0,0};
	this->trajectoryCmd.velocities = {0,0,0,0,0};
	this->trajectoryCmd.accelerations = {0,0,0,0,0};
	this->jointState.position = {0,0,0,0,0};
	this->jointState.velocity = {0,0,0,0,0};
	this->jointState.effort = {0,0,0,0,0};
	this->currentFbk.trajectoryCmd.positions = {0,0,0,0,0};
	this->currentFbk.trajectoryCmd.velocities = {0,0,0,0,0};
	this->currentFbk.trajectoryCmd.accelerations = {0,0,0,0,0};
	this->currentFbk.jointState.position = {0,0,0,0,0};
	this->currentFbk.jointState.velocity = {0,0,0,0,0};
	this->currentFbk.jointState.effort = {0,0,0,0,0};
	this->currentFbk.deflections = {0,0,0,0,0};
	this->currentFbk.velocityFlt = {0,0,0,0,0};
	this->currentFbk.deflection_vel = {0,0,0,0,0};
	this->currentFbk.motorSensorTemperature = {0,0,0,0,0};
	this->currentFbk.windingTemp = {0,0,0,0,0};
	this->currentFbk.windingTempFlt = {0,0,0,0,0};
	this->currentFbk.torqueCmd = {0,0,0,0,0};
	this->currentFbk.torqueID = {0,0,0,0,0};
	this->currentFbk.accel = {0,0,0,0,0};
	this->currentFbk.epsTau = {0,0,0,0,0};

	this->lastCommand_time = ros::WallTime::now();
}

void armcontroller::init()
{
	// vec_module.clear();
	// vec_module.resize(5);
	// Setup the lookup
	long timeout_ms = 5000; // Give the modules plenty of time to appear.
	float freq_hz = 1000;
	hebi::Lookup lookup;
	std::vector<hebi::MacAddress> macs;
	
	// Ky:Commented out IMR Arm 1 Mac Addresses
	/*(std::vector<std::string> modules = {"d8:80:39:65:ae:44",
																			"d8:80:39:9d:64:fd",
																			"d8:80:39:9d:59:c7",
																			"d8:80:39:9d:4b:cd",
																			"d8:80:39:9c:d7:0d"};*/

	// Ky: Adding in test research arm
	std::vector<std::string> modules = {"D8:80:39:E8:B3:3C",
																			"D8:80:39:E9:06:36",
																			"D8:80:39:9D:29:4E",
																			"D8:80:39:9D:3F:9D",
																			"D8:80:39:9D:04:78"};

	//Build the vector of mac addresses anc check for correct
	//hex strings
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
	hebi::GroupFeedback groupFeedback(this->arm_group->size());
	bool fbkSuccess = true;

	//Keep the timout ms at 5. This is a tradeoff between the delay being too long and blocking the sequence or
	//being too short and dropping too many packets because they are not received quickly enough
	if(!arm_group->requestFeedback(&groupFeedback,5)){fbkSuccess = false;}
	if(fbkSuccess)
	{
		ros::WallTime time_stamp = ros::WallTime::now();
		this->currentFbk.header.stamp.sec = time_stamp.sec;
		this->currentFbk.header.stamp.nsec = time_stamp.nsec;
		this->jointState.header.stamp.sec = time_stamp.sec;
		this->jointState.header.stamp.nsec = time_stamp.nsec;
		this->jointState.name = {std::string("Joint1"),std::string("Joint2"),std::string("Joint3"),std::string("Joint4")};

		std::vector<double> velocityUpdate(5);
		std::vector<double> windingTempUpdate(5);

		for(int i=0;i<5;i++)
		{
			this->jointState.position[i] = groupFeedback[i].actuator().position().get();
			this->jointState.velocity[i] = groupFeedback[i].actuator().velocity().get();
			this->jointState.effort[i] = groupFeedback[i].actuator().torque().get();

			this->currentFbk.jointState.position[i] = groupFeedback[i].actuator().position().get();
			this->currentFbk.jointState.velocity[i] = groupFeedback[i].actuator().velocity().get();
			this->currentFbk.jointState.effort[i] = groupFeedback[i].actuator().torque().get();

			if(!this->initialize)
			{
				this->initial_position = this->jointState.position;
				ros::WallTime prevTime = ros::WallTime::now();
			}

			this->trajectoryCmd.positions[i] = this->currentCmd.jointTrajectory.points[0].positions[i];
			this->trajectoryCmd.velocities[i] = this->currentCmd.jointTrajectory.points[0].velocities[i];
			this->trajectoryCmd.accelerations[i] = this->currentCmd.jointTrajectory.points[0].accelerations[i];

			this->currentFbk.trajectoryCmd.positions[i] = this->currentCmd.jointTrajectory.points[0].positions[i];
			this->currentFbk.trajectoryCmd.velocities[i] = this->currentCmd.jointTrajectory.points[0].velocities[i];
			this->currentFbk.trajectoryCmd.accelerations[i] = this->currentCmd.jointTrajectory.points[0].accelerations[i];

			this->currentFbk.epsTau[i] = this->currentCmd.epsTau[i];

			this->currentFbk.torqueCmd[i] = groupFeedback[i].actuator().torqueCommand().get();
			this->currentFbk.deflections[i] = groupFeedback[i].actuator().deflection().get();
			this->currentFbk.windingTemp[i] = groupFeedback[i].actuator().motorWindingTemperature().get();
			this->currentFbk.motorSensorTemperature[i] = groupFeedback[i].actuator().motorSensorTemperature().get();

			velocityUpdate[i] = this->jointState.velocity[i];
			windingTempUpdate[i] = this->currentFbk.windingTemp[i];
		}

		if(!this->initialize)
		{
			std::cout << "Modules Fully Initialized" << std::endl;
			this->initialize = true;
		}
		
		std::vector<double> velocityFiltered(5);
		std::vector<double> windingTempFiltered(5);
		WMAFilter(velocityUpdate, windingTempUpdate, velocityFiltered, windingTempFiltered);

		Eigen::VectorXd positionMeas(5);
		Eigen::VectorXd velocityMeas(5);
		Eigen::VectorXd accelMeas(5);
		
		double fc = 1;
		double dt = (time_stamp-this->prevTime).toSec();
		double alpha_accel = 1/(1+dt*2*M_PI*fc);

		fc = 1;;
		double alpha_def = 1/(1+dt*2*M_PI*fc);

		for(int i=0;i<5;i++)
		{
			 positionMeas[i] = this->currentFbk.jointState.position[i];
			 velocityMeas[i] = this->currentFbk.jointState.velocity[i];
			 this->currentFbk.deflection_vel[i] = alpha_def*this->currentFbk.deflection_vel[i] +(1-alpha_def)*(this->currentFbk.deflections[i]-prevDeflection[i])/dt;
			 this->currentFbk.accel[i] = alpha_accel*this->currentFbk.accel[i] +(1-alpha_accel)*(velocityFiltered[i]-prevVelFlt[i])/dt;
			 accelMeas[i] = this->currentFbk.accel[i];
			 prevDeflection[i] = this->currentFbk.deflections[i];
			 prevVelFlt[i] = velocityFiltered[i];
		}
		std::vector<double> torqueID(5);
		dynamics::inverseDynamics(positionMeas,velocityMeas,accelMeas,torqueID);
		
		for(int i=0;i<5;i++)
		{
			this->currentFbk.velocityFlt[i] = velocityFiltered[i];
			this->currentFbk.windingTempFlt[i] = windingTempFiltered[i];
			this->currentFbk.torqueID[i] = torqueID[i];
		}
		this->prevTime = time_stamp;
	}
	
	// if(fbkSuccess == false){
	//   // printf("Did not receive feedback!\n");
	//   this->fbkfailure_count++;
	// }
	// else
	// {
	//   // printf("Got feedback.\n");
	//   this->fbksuccess_count++;
	// }
	// double failure_rate = 100*fbkfailure_count/(fbkfailure_count+fbksuccess_count);
	// std::cout << "Failure_Rate" << failure_rate << std::endl;
	return fbkSuccess;
}

void armcontroller::getFeedbackMsg(sensor_msgs::JointState &jointState_fbk,
																	trajectory_msgs::JointTrajectoryPoint &trajectoryCmd_fbk,
																	model_learning::FeedbackML &armcontroller_fbk)
{
	jointState_fbk = this->jointState;
	trajectoryCmd_fbk = this->trajectoryCmd;
	armcontroller_fbk = this->currentFbk;
}

void armcontroller::controller(const Eigen::VectorXd &positionCmd,const Eigen::VectorXd &velocityCmd,
						 const Eigen::VectorXd &accelCmd, std::vector<double> &torque)
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
	error_pos = positionCmd-positionFB;
	error_vel = velocityCmd-velocityFB;

	for(int i=0;i<5;i++)
	{
		posFbk[i] = positionFB[i];
	}

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

		if(feedforward)
		{
			//dynamics::inverseDynamics(positionCmd,velocityCmd,accelCmd,torqueID);
			dynamics::inverseDynamics(positionCmd,velocityCmd,accelCmd,torqueID);
		}
		else
		{
			dynamics::gravityComp(posFbk,torqueID);
		}
		
		if (closedLoop)
		{
			feedbackControl(error_pos,error_vel,torqueFB);
		}
		else
		{
			torqueFB = {0,0,0,0,0};
		}


		for(uint i=0;i<5;i++)
		{
			torque[i] = torqueID[i] + torqueFB[i];

			this->currentCmd.jointTrajectory.points[0].positions[i] = positionCmd[i];
			this->currentCmd.jointTrajectory.points[0].velocities[i] = velocityCmd[i];
			this->currentCmd.jointTrajectory.points[0].accelerations[i] = accelCmd[i];
		}
	}
}

void armcontroller::feedbackControl(const Eigen::VectorXd &error_pos,const Eigen::VectorXd &error_vel,
								std::vector<double> &torque)
{
	Eigen::MatrixXd Kp(5,5);
	Eigen::MatrixXd Kv(5,5);
	Eigen::VectorXd torqueFB(5);
	Eigen::VectorXd vec_p(5);
	Eigen::VectorXd vec_v(5);

	for(int i=0;i<5;i++)
	{
		vec_p[i] = this->currentCmd.pos_gain[i];
		vec_v[i] = this->currentCmd.vel_gain[i];
	}

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

	torqueFB = Kp*error_pos + Kv*error_vel;
	for(uint i=0;i<5;i++)
	{
		torque[i] = torqueFB[i];
	}
}

bool armcontroller::sendCommand(const std::vector<double> &alpha, const std::vector<double> &torque)   //TODO: Cap the output rate at 500Hz
{
	hebi::GroupCommand command(this->arm_group->size());
	bool bSuccess = true;
	for(uint j = 0; j < 5; j++)
	{
		command[j].actuator().position().set(alpha[j]);
		command[j].actuator().torque().set(torque[j]);
		// cmd.actuator().position().set(alpha[j]);
		// cmd.actuator().torque().set(torque[j]);
		// printf("send command to joint %d with angle %f\n", j, alpha[j]);
		// printf("send command to joint %d with torque %f\n", j, torque[j]);
		// if(!vec_module[j]->sendCommand(cmd))
		// {
		//   bSuccess = false;
		// }
		// hebi_sleep_ms(100);
	}
	if(!arm_group->sendCommand(command)){bSuccess = false;}
	if(bSuccess == false){
		// printf("Did not receive acknowledgement!\n");
	}else{
		// printf("Got acknowledgement.\n");
	}
	return bSuccess;
}

bool armcontroller::sendTorqueCommand(const std::vector<double> &torque)   //TODO: Cap the output rate at 500Hz
{
	bool bSuccess = true;
	hebi::GroupCommand command(this->arm_group->size());
	// ros::WallTime currentCmd_time = ros::WallTime::now();
	// double millisec = (currentCmd_time-this->lastCommand_time).toNSec()*1e-6;
	// this->lastCommand_time = currentCmd_time;
	// if(millisec < 2.)
	// {
	//   bSuccess = false;
	//   return bSuccess;
	// }

	for(uint j = 0; j < 5; j++)
	{
		command[j].actuator().torque().set(torque[j]);
		// cmd.actuator().torque().set(torque[j]);
		// if(!vec_module[j]->sendCommand(cmd))
		// {
		//   bSuccess = false;
		// }
	}
	if(!arm_group->sendCommand(command)){bSuccess = false;}
	if(bSuccess == false){
		// printf("Did not receive acknowledgement!\n");
	}else{
		// printf("Got acknowledgement.\n");
	}
	return bSuccess;
}

void armcontroller::WMAFilter(const std::vector<double> &velocity, const std::vector<double> &motorTemp,
											 std::vector<double> &velocityFlt, std::vector<double> &motorTempFlt)
{
	this->single_vel_que1.push_back(velocity[0]);
	this->single_vel_que2.push_back(velocity[1]);
	this->single_vel_que3.push_back(velocity[2]);
	this->single_vel_que4.push_back(velocity[3]);
	this->single_vel_que5.push_back(velocity[4]);

	if(this->single_vel_que1.size()>(this->num_vel_filt+1))
	{
		this->single_vel_que1.pop_front();
		this->single_vel_que2.pop_front();
		this->single_vel_que3.pop_front();
		this->single_vel_que4.pop_front();
		this->single_vel_que5.pop_front();
	}
	int vel_sum;
	if(this->single_vel_que1.size()==this->num_vel_filt+1)
	{
		vel_sum = (this->single_vel_que1.size()*(this->single_vel_que1.size()-1))/2;
	}
	else
	{
		vel_sum = (this->single_vel_que1.size()*(this->single_vel_que1.size()+1))/2;
	}
	
	movingAverage({this->single_vel_que1.begin(),this->single_vel_que1.end()},
								 this->num_vel_filt,vel_sum,this->WMA_vel_single[0],this->numerator_vel_single[0],this->total_vel_single[0]);
	movingAverage({this->single_vel_que2.begin(),this->single_vel_que2.end()},
								 this->num_vel_filt,vel_sum,this->WMA_vel_single[1],this->numerator_vel_single[1],this->total_vel_single[1]);
	movingAverage({this->single_vel_que3.begin(),this->single_vel_que3.end()},
								 this->num_vel_filt,vel_sum,this->WMA_vel_single[2],this->numerator_vel_single[2],this->total_vel_single[2]);
	movingAverage({this->single_vel_que4.begin(),this->single_vel_que4.end()},
								 this->num_vel_filt,vel_sum,this->WMA_vel_single[3],this->numerator_vel_single[3],this->total_vel_single[3]);
	movingAverage({this->single_vel_que5.begin(),this->single_vel_que5.end()},
								 this->num_vel_filt,vel_sum,this->WMA_vel_single[4],this->numerator_vel_single[4],this->total_vel_single[4]);

	this->single_temp_que1.push_back(motorTemp[0]);
	this->single_temp_que2.push_back(motorTemp[1]);
	this->single_temp_que3.push_back(motorTemp[2]);
	this->single_temp_que4.push_back(motorTemp[3]);
	this->single_temp_que5.push_back(motorTemp[4]);

	if(this->single_temp_que1.size()>(this->num_temp_filt+1))
	{
		this->single_temp_que1.pop_front();
		this->single_temp_que2.pop_front();
		this->single_temp_que3.pop_front();
		this->single_temp_que4.pop_front();
		this->single_temp_que5.pop_front();
	}
	int temp_sum;
	if(this->single_temp_que1.size()==this->num_temp_filt+1)
	{
		temp_sum = (int)(this->single_temp_que1.size()*(this->single_temp_que1.size()-1))/2;
	}
	else
	{
		temp_sum = (int)(this->single_temp_que1.size()*(this->single_temp_que1.size()+1))/2;
	}
	movingAverage({this->single_temp_que1.begin(),this->single_temp_que1.end()},
								 this->num_temp_filt,temp_sum,this->WMA_temp_single[0],this->numerator_temp_single[0],this->total_temp_single[0]);
	movingAverage({this->single_temp_que2.begin(),this->single_temp_que2.end()},
								 this->num_temp_filt,temp_sum,this->WMA_temp_single[1],this->numerator_temp_single[1],this->total_temp_single[1]);
	movingAverage({this->single_temp_que3.begin(),this->single_temp_que3.end()},
								 this->num_temp_filt,temp_sum,this->WMA_temp_single[2],this->numerator_temp_single[2],this->total_temp_single[2]);
	movingAverage({this->single_temp_que4.begin(),this->single_temp_que4.end()},
								 this->num_temp_filt,temp_sum,this->WMA_temp_single[3],this->numerator_temp_single[3],this->total_temp_single[3]);
	movingAverage({this->single_temp_que5.begin(),this->single_temp_que5.end()},
								 this->num_temp_filt,temp_sum,this->WMA_temp_single[4],this->numerator_temp_single[4],this->total_temp_single[4]);
	for(int i=0; i<5; i++)
	{
			velocityFlt[i] = this->WMA_vel_single[i];
			motorTempFlt[i] = this->WMA_temp_single[i];
	}
};

void armcontroller::movingAverage(const std::vector<double> &data_vector,const int &num,const int &vec_sum,
									double &WMA,double &numerator,double &total) const
{
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
	// std::chrono::milliseconds ms_start,ms_current,ms_dt;
	std::vector<double> torque(5);
	std::vector<double> alpha(5);
	
	// std::vector<double> initial_position(5);
	Eigen::VectorXd positionCmd(5);
	Eigen::VectorXd velocityCmd(5);
	Eigen::VectorXd accelCmd(5);

	trajectory_msgs::JointTrajectory jointTrajectory = cmd.jointTrajectory;
	double motorOn = cmd.motorOn;
	double cType = cmd.controlType;
	// double dtList = cmd.dtList;
	std::vector<double> epsTau = cmd.epsTau;
	this->currentCmd.closedLoop = cmd.closedLoop;
	this->currentCmd.feedforward = cmd.feedforward;
	this->currentCmd.pos_gain = cmd.pos_gain;
	this->currentCmd.vel_gain = cmd.vel_gain;
	
			// ms_start = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
	// ms_current = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch())-ms_start;

	this->controlType = cType;
	for(uint j=0;j<5;j++)
	{
		if(cType == 0.)
		{
			positionCmd[j] = jointTrajectory.points[0].positions[j];
			alpha[j] = positionCmd[j];
		}
		else if(cType == 1.)
		{
			positionCmd[j] = jointTrajectory.points[0].positions[j];//+this->initial_position[j];
			velocityCmd[j] = jointTrajectory.points[0].velocities[j];
			accelCmd[j] = jointTrajectory.points[0].accelerations[j];
		}
	}
	controller(positionCmd,velocityCmd,accelCmd,torque);
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
	if(motorOn && cType == 1.)
	{
		sendTorqueCommand(torque);

	}
	else if(motorOn && cType == 0.)
	{
		sendCommand(alpha,torque);
	}
	
}

int main(int argc, char* argv[])
{
	armcontroller ac;
	ac.init();

	ros::init(argc, argv, "arm_1_traj_node");

	ros::NodeHandle nh;
	ros::Publisher jointState_pub = nh.advertise<sensor_msgs::JointState>("jointState_fbk",10);
	ros::Publisher trajectory_pub = nh.advertise<trajectory_msgs::JointTrajectoryPoint>("trajectoryCmd_fbk",10);
	ros::Publisher armcontroller_pub = nh.advertise<model_learning::FeedbackML>("armcontroller_fbk",10);
	ros::Subscriber armcontroller_sub = nh.subscribe("ml_publisher",10,&armcontroller::subscriberCallback, &ac);
	ros::Rate loop_rate(100);

	// ros::ServiceClient cli = nh.serviceClient
	// //   <arm_planner::arm_planning>("arm_planner/arm_planning_1");

	// ros::ServiceClient cli = nh.serviceClient<arm_planner::arm_planning>("reset_arm_position");

	// arm_planner::arm_planning armSrv;

	// init arm controller
	

	// string topicName_traj_1;
	std::cout << "enter ros spin" << std::endl;

	// if ( !nh.getParam("arm_controller/topic_name_arm_1_traj", topicName_traj_1))
	//     cout << "FAIL TO GET arm_controller/topic_name_arm_1_traj" << endl;

	//   ros::ServiceServer server_traj = nh.advertiseService(topicName_traj_1.data(), 
	//          &armcontroller::serviceCallback, &ac);

	double mins = 999.;    
	while(1)//ros::ok() %Checking if ros is ok slows down the loop to ~87 Hz
	{
		ros::WallTime start = ros::WallTime::now();
		sensor_msgs::JointState jointState_fbk;
		trajectory_msgs::JointTrajectoryPoint trajectoryCmd_fbk;
		model_learning::FeedbackML armcontroller_fbk;

		if(ac.updateFeedback())
		{
			ac.getFeedbackMsg(jointState_fbk,trajectoryCmd_fbk,armcontroller_fbk);
			jointState_pub.publish(jointState_fbk);
			trajectory_pub.publish(trajectoryCmd_fbk);
			armcontroller_pub.publish(armcontroller_fbk);
		}
		

		// if(dt > ac.maxDt)
		// {
		//   ac.maxDt = dt;
		// }

		ros::spinOnce();
		// ros::WallTime end = ros::WallTime::now();
		// double freqs = 1/(end-start).toSec();
		// std::cout << mins << std::endl;
		// if(freqs<mins)
		// {
		//   mins = freqs;
		// }
		loop_rate.sleep();
	}

	return 0;

}


