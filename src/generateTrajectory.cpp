 


bool generateTrajectory(const geometry_msgs::PoseArray &taskSpaceTrajectory,
            trajectory_msgs::JointTrajectory &jointSpaceTrajectory)
{
    std::vector<double> theta(5);

    for(int=0,i<taskSpaceTrajectory.points.size(),i++)
    {
        if(inverseKinematics(taskSpaceTrajectory.points[i],theta)
        {

        }
    }
}


 bool inverseKinematics(const geometry_msgs::Pose &pose, std::vector<double> &alpha){
    Quaternion<double> q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    #convert pose (quaterion) to matrix
    Matrix4d g1i6;
    g1i6.setIdentity();
    g1i6.block(0, 0, 3, 3) << q.matrix();
    g1i6.block(0, 3, 3, 1) << pose.position.x, pose.position.y, pose.position.z;
    #cout << "desired end-effector pose:" << endl << g1i6 << endl;
    #make sure g1i6 is pointing downwards
    if(g1i6(2, 2) > -0.95){
      std::cout << "Invalid end-effector pose. Make sure the end-effector is pointing downwards." << std::endl;
      return false;
    }
    # Matrix4d g4o6 = T4o5i * rotationZ(cmdAngle[4]) * T5o6;
    Vector3d xAxis = g1i6.block(0, 0, 3, 1).normalized();
    # cout << "atan2(orientation) " << atan2(g1i6(1, 3), g1i6(0, 3)) << endl;
    alpha[0] = atan2(g1i6(1, 3), g1i6(0, 3)) + asin(4.5 / sqrt(g1i6(0, 3) * 10000.0 * g1i6(0, 3)  + g1i6(1, 3) * 10000.0 * g1i6(1, 3)));
    #cout << "alpha[0] " << alpha[0] << endl;
    alpha[4] = M_PI - atan2(g1i6(1, 0), g1i6(0, 0)) + alpha[0];
    #cout << "alpha[4] before adjust " << alpha[4] << endl;
    if(last_alpha4 > -100.0){
      double temp1 = alpha[4]+2*M_PI;
      double temp2 = alpha[4]-2*M_PI;
      if( fabs(temp1 - last_alpha4) < 0.5 )
        alpha[4] = temp1;
      if( fabs(temp2 - last_alpha4) < 0.5 )
        alpha[4] = temp2;
    }
    last_alpha4 = alpha[4];
    # << "alpha[4] after adjust " << alpha[4] << endl;
    Matrix4d g4o6 = T4o5i * rotationZ(alpha[4]) * T5o6;
    #get the g1i4o
    Matrix4d g1i4o = g1i6 * g4o6.inverse();
    #using simple inverse kinematics to  compute joint angles 1, 2, 3
    # alpha[0] = atan2(g1i4o(1, 3), g1i4o(0, 3));
    
    Matrix4d g1i2i = rotationZ(alpha[0]) * T1o2i;
    Matrix4d g2i4o = g1i2i.inverse() * g1i4o;