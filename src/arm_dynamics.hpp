//! Dynamics functions for the HEBI Robotics 5-DOF Manipulator Arm
/*!Calculate the dynamics equations of a 5-DOF HEBI manipulator
    arm used on the Biorobotics Lab Intelligent Mobile Robot (IMR) project

    @Author Ky Woodard
    @version 1.0 2/12/2017
 */

#ifndef ARM_DYNAMICS_HPP_
#define ARM_DYNAMICS_HPP_

#include <eigen3/Eigen/Eigen>

namespace dynamics {
	/*! Calculate the necessary torque required to cancel
						 gravitational forces on the joints
		@param[in] theta current angles of the five joints [rad]
		@param[out] torque computed torque on the 5 joints [N-m]
	*/
	void gravityComp(const std::vector<double> & theta, std::vector<double> & torque);
	/*!Calculate the inverse dynamics of the arm, which
					 	 includes inertia, corriolis, gravitational, and
					 	 friction effect
		@param[in] theta current angle of the 5 joints [rad]
		@param[in] omega current angular velocity of the 5 joints [rad/s]
		@param[in] alpha desired angular acceleration of the 5 joints [rad/s^2]
		@param[out] torque computed torque to produce desired acceleration
						 at the 5 joints [N-m]
	*/
	void inverseDynamics(const Eigen::VectorXd& theta,const Eigen::VectorXd& omega,
						  const Eigen::VectorXd& alpha,std::vector<double>& torque);
	/*!Calculate the gravity torque of the inverse dynamics
						 based on the current angle of the joints
		@param[in] theta current angle of the 5 joints [rad]
		@param[out] torque computed torque on the 5 joints [N-m]
	*/
	static void gravity(const Eigen::VectorXd& theta,Eigen::VectorXd& torque);
	/*!Calculate the inertia torque of the inverse dynamics
						 based on the current angle of the joints
		@param[in] theta current angle of the 5 joints [rad]
		@param[out] Inertia inertia matrix (5x5)
	*/
	static void inertia(const Eigen::VectorXd &theta,Eigen::MatrixXd &Inertia);
	/*!Calculate the corriolis torque of the inverse dynamics
			based on the current angle and angular velocity of the joints
		@param[in] theta current angle of the 5 joints [rad]
		@param[in] omega current angle of the 5 joints [rad/s]
		@param[out] Corriolis corriolis matrix (5x5)
	*/
	static void corriolis(const Eigen::VectorXd& theta,const Eigen::VectorXd& omega,
					Eigen::MatrixXd& corrioslis);
} //dynamics namespace


#endif // ARM_DYNAMICS_HPP_