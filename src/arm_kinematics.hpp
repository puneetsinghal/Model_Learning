#ifndef ARM_KINEMATICS_HPP_
#define ARM_KINEMATICS_HPP_

#include <eigen3/Eigen/Eigen>

namespace kinematics {
void forwardKinematics(const std::vector<double>&, Eigen::Matrix4d&);
bool inverseKinematics(const geometry_msgs::Pose&, std::vector<double> &);
} //kinematics namespace


#endif // ARM_KINEMATICS_HPP_