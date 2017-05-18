//!Kinematics functions for the HEBI Robotics 5-DOF Manipulator Arm
/*!Calculate the kinematic equations of a 5-DOF HEBI manipulator
		arm used on the Biorobotics Lab Intelligent Mobile Robot (IMR) project

		@Author Chaohui Gong (edited: Ky Woodard)
 */

#ifndef ARM_KINEMATICS_HPP_
#define ARM_KINEMATICS_HPP_

#include <eigen3/Eigen/Eigen>
#include "geometry_msgs/Pose"

namespace kinematics
{
	/*! Compute the forward kinematics of the robotic arm for the given joint angles
		@param[in] alpha current angles of the five joints [rad]
		@param[out] gli6 homogenous transformation from the base to the end effector(4x4)
	*/
	bool forwardKinematics(const std::vector<double> &alpha,Eigen::Matrix4d &g1i6);
	/*! Compute the inverse kinematics of the robotic arm for the given joint angles
		@param[in] pose desired pose of the end effector in the base frame [Pose]
		@param[out] alpha joint angles of the five joints [rad]
	*/
	void inverseKinematics(const geometry_msgs::Pose &pose,std::vector<double> &alpha);
} //kinematics namespace


#endif // ARM_KINEMATICS_HPP_