//! Dynamics functions for the HEBI Robotics 5-DOF Manipulator Arm
/*!Calculate the dynamics equations of a 5-DOF HEBI manipulator
    arm used on the Biorobotics Lab Intelligent Mobile Robot (IMR) project

    @Author Ky Woodard
    @version 1.0 2/12/2017
 */

    Functions:
		gravityComp
			Description: Calculate the necessary torque required to cancel
						 gravitational forces on the joints
			Input:
				theta  - current angles of the five joints [rad]
			Output:
				torque - computed torque on the 5 joints [N-m]

		inverseDynamics
			Description: 
			Inputs:
				theta  - current angle of the 5 joints [rad]
				omega  - current angular velocity of the 5 joints [rad/s]
				alpha  - desired angular acceleration of the 5 joints [rad/s^2]
			Output:
				torque - computed torque to produce desired acceleration
						 at the 5 joints [N-m]

		gravity
			Description: 
			Inputs:
				theta  - current angle of the 5 joints
			Output:
				torque - computed torque on the 5 joints [N-m]

		inertia
			Description: Calculate the inertia torque of the inverse dynamics
						 based on the current angle of the joints
			Inputs:
				theta  - current angle of the 5 joints [rad]
			Output:
				Inertia - inertia matrix (5x5)

		corriolis
			Description: Calculate the corriolis torque of the inverse dynamics
						 based on the current angle and angular velocity of the
						 joints
			Inputs:
				theta  - current angle of the 5 joints [rad]
				omega  	- current angle of the 5 joints [rad/s]
			Output:
				Corriolis - Corriolis matrix (5x5)
*/

#ifndef ARM_DYNAMICS_HPP_
#define ARM_DYNAMICS_HPP_

#include <eigen3/Eigen/Eigen>

namespace dynamics {
	/*! Calculate the necessary torque required to cancel
						 gravitational forces on the joints
	*/
	void gravityComp(const std::vector<double> &, std::vector<double> &);
	/*!Calculate the inverse dynamics of the arm, which
					 	 includes inertia, corriolis, gravitational, and
					 	 friction effect
	*/
	void inverseDynamics(const Eigen::VectorXd &,const Eigen::VectorXd &,
						  const Eigen::VectorXd &,std::vector<double> &);
	/*!Calculate the gravity torque of the inverse dynamics
						 based on the current angle of the joints
	*/
	static void gravity(const Eigen::VectorXd &,Eigen::VectorXd &);
	/*!Calculate the inertia torque of the inverse dynamics
						 based on the current angle of the joints
	*/
	static void inertia(const Eigen::VectorXd &,Eigen::MatrixXd &);
	/*Calculate the corriolis torque of the inverse dynamics
						 based on the current angle and angular velocity of the
						 joints
	*/
	static void corriolis(const Eigen::VectorXd &,const Eigen::VectorXd &,
					Eigen::MatrixXd &);
} //dynamics namespace


#endif // ARM_DYNAMICS_HPP_