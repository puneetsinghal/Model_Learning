#ifndef ARM_KINEMATICS_HPP_
#define ARM_KINEMATICS_HPP_

#include <eigen3/Eigen/Eigen>
#include "geometry_msgs/Pose"

namespace kinematics
{
	bool inverseKinematics(const geometry_msgs::Pose &pose,
											std::vector<double> &alpha)
	{
		Eigen::Quaternion<double> q(pose.orientation.w, pose.orientation.x, 
								pose.orientation.y, pose.orientation.z);
		//convert pose (quaterion) to matrix
		Eigen::Matrix4d g1i6;
		g1i6.setIdentity();
		g1i6.block(0, 0, 3, 3) << q.matrix();
		g1i6.block(0, 3, 3, 1) << pose.position.x, pose.position.y, pose.position.z;
		//make sure g1i6 is pointing downwards
		if(g1i6(2, 2) > -0.95){
			std::cout << "Invalid end-effector pose. Make sure the end-effector is pointing downwards." << std::endl;
			return false;
		}
		// Matrix4d g4o6 = T4o5i * rotationZ(cmdAngle[4]) * T5o6;
		Eigen::Vector3d xAxis = g1i6.block(0, 0, 3, 1).normalized();
		alpha[0] = atan2(g1i6(1, 3), g1i6(0, 3)) + asin(4.5 / sqrt(g1i6(0, 3) * 10000.0 * g1i6(0, 3)  + g1i6(1, 3) * 10000.0 * g1i6(1, 3)));
		alpha[4] = M_PI - atan2(g1i6(1, 0), g1i6(0, 0)) + alpha[0];
		if(last_alpha4 > -100.0){
			double temp1 = alpha[4]+2*M_PI;
			double temp2 = alpha[4]-2*M_PI;
			if( fabs(temp1 - last_alpha4) < 0.5 )
				alpha[4] = temp1;
			if( fabs(temp2 - last_alpha4) < 0.5 )
				alpha[4] = temp2;
		}
		last_alpha4 = alpha[4];
		Eigen::Matrix4d g4o6 = T4o5i * rotationZ(alpha[4]) * T5o6;
		//get the g1i4o
		Eigen::Matrix4d g1i4o = g1i6 * g4o6.inverse();
		//using simple inverse kinematics to  compute joint angles 1, 2, 3
		// alpha[0] = atan2(g1i4o(1, 3), g1i4o(0, 3));
		
		Eigen::Matrix4d g1i2i = rotationZ(alpha[0]) * T1o2i;
		Eigen::Matrix4d g2i4o = g1i2i.inverse() * g1i4o;

		double elevation = g2i4o(1, 3), reach = g2i4o(0, 3);
		double L2 = 0.280;
		double L3 = sqrt(pow(elevation, 2) + pow(reach, 2));
		if((L3 * L3 / L2 / L2 - 2.0 < -2.0) || (L3 * L3 / L2 / L2  - 2.0 > 2.0)){
			return false;
		}else{
			alpha[2] = acos(((pow(elevation, 2) + pow(reach, 2)) / L2 / L2 - 2.0) / 2.0);
		}
		if((L2 / L3 * sin(alpha[2])) < -1 || (L2 / L3 * sin(alpha[2])) > 1){
			return false;
		}
		double beta = asin(L2 / L3 * sin(alpha[2]));
		alpha[1] = atan2(elevation, reach) + beta - M_PI / 2.0;
		alpha[3] = M_PI - alpha[1] + alpha[2];
		//printf("alpha3=%f\n", alpha[3]);
		// gl2 = rotationZ(alpha[0])*Toff*T1oCi;
		// gl3 = g1i2i * rotationZ(alpha[1]) * Toff * T2oCi;
		// gl4 = g1i2i * rotationZ(alpha[1]) * Toff * T2o3i * rotaionZ(alpha[2]) * Toff * T3oCi;
		// gl5 = (g1i2i * rotationZ(alpha[1]) * Toff * T2o3i * rotaionZ(alpha[2]) * Toff * T3o4i
		//        * rotationZ(alpha[3]) * Toff * T4oCi)
		// forwardKinematics(alpha);
		return true;
	}

	void forwardKinematics(const std::vector<double> &alpha,
											Eigen::Matrix4d &g1i6)
	{
		g1i6 = rotationZ(alpha[0]) * T1o2i * rotationZ(alpha[1]) * 
			T2o3i * rotationZ(alpha[2]) *
			T3o4i * rotationZ(alpha[3]) * 
			T4o5i * rotationZ(alpha[4]) * T5o6;
	}
}
#endif