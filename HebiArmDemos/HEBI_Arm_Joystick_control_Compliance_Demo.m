%%This Script Run a Demo of the 5-DOF arm in the biorobotics lab capable of
%This particular demo controls the arm using a standard Joystick. This demo
%includes impedance control
%--------------------------------------------------------------------------
%
%Zero position of arm is considered with the arm fully upright. Total
%length from top of 1st X-module should be approximately 66.5cm
%
%
%
%Revision History
%       Date        Person responsible      Notes:
%-----------------------------------------------------------------------
%1. 28-Feb-2017         Kaveh Nikou         Creation file
%
%
%-------------------------------------------------------------------------
%%
%clear plots and workspace
close all
clc
clear all

%Global variables
gravity = [0 0 1];
F = 0;
F1 = 0;
F2 = 0;
F3 = 0;

%%
%1. Add the path of the Hebi SDK and Plotting tools to the workspace
addpath(genpath('C:\Users\Johnsg7\Desktop\CMU Internship\Model Learning\Computer Vision\Matlab demos\Hebi Arm Demos\hebi'));
addpath(genpath('C:\Users\Johnsg7\Desktop\CMU Internship\Model Learning\Computer Vision\Matlab demos\Hebi Arm Demos\matlab_SEA'));
addpath(genpath('C:\Users\Johnsg7\Desktop\CMU Internship\Model Learning\Computer Vision\Matlab demos\Hebi Arm Demos\Utilities'));
%%
%2. Setup the kinematics configuration of the 5-DOF arm
HebiKinematicsSetup;
n = group.getNumModules;
%%
%3. set gains and limits
%setGains;

%4. create command structure
cmd = CommandStruct();

%5. create joystick object
joy = vrjoystick(1);

%%
%6. Get initial feedback from modules as state machines initial state
fbk = group.getNextFeedback();
endeffector = kin.getForwardKinematics('EndEffector',fbk.position);
xyz = endeffector(1:3,4);
so3 = endeffector(1:3,1:3);

%initially disable all motors
cmd.position = nan(1,n);
cmd.velocity = nan(1,n);
cmd.torque = nan(1,n);
group.set(cmd);
pause(1);

group.setFeedbackFrequency(100);
%7.Set up plotting tool
links = plotHebiManipulator;
plt = HebiPlotter('JointTypes', links,'resolution','low');


%%
%9. begin the state loop
ok = 1;
tic
while ok
    time = toc;
    %get module feedback
    fbk = group.getNextFeedback();

     %initialize joystick buttons and map joystick to coordinates
     [axes,buttons,povs] = read(joy); 
     axes(abs(axes)<.2) = 0;   
     ok= ~buttons(10);

    %calculate the numerical Jacobian
    J = kin.getJacobian('endEffector',fbk.position);
    
    F3 = F2;
    F2 = F1;
    F1 = F;
    F = -(pinv(J'))*(fbk.torque-kin.getGravCompTorques(fbk.position,gravity))';

    
    weighted_av = (4.*F + 3.*F1 + 2.*F2 + F3)./10;
    weighted_av(3)
    if weighted_av(3) >=5
      xyz_prev = xyz;  %save previous value of position
      if axes(4) < -0.3
       xyz = xyz + [-axes(1) axes(2)  0]'*0.002;
      else
       xyz = xyz + [-axes(1) axes(2)  axes(4)]'*0.002;   
      end
      %calculate the numerical Jacobian
      J = kin.getJacobian('endEffector',fbk.position);
    
      F3 = F2;
      F2 = F1;
      F1 = F;
      F = -(pinv(J'))*(fbk.torque-kin.getGravCompTorques(fbk.position,gravity))';

    
    weighted_av = (4.*F + 3.*F1 + 2.*F2 + F3)./10;
    weighted_av(3)
    else
     xyz_prev = xyz;  %save previous value of position
     xyz = xyz + [-axes(1) axes(2)  axes(4)]'*0.002;
    end
    %set gravity compensation and position commands
    cmd.torque = kin.getGravCompTorques(fbk.position,gravity);
    so3 = ones(3);
    so3(3,3) = -1;
    positions = kin.getInverseKinematics('xyz',xyz,'TipAxis',...
                        [0 0 -1],'InitialPositions',fbk.position,...
                        'MaxIterations',10000);
    cmd.position = positions;
    group.set(cmd);
    
    %plt.plot(positions); %disabled becasue plotting slows down the
    %processing significantly
    toc - time; %calculate the processing time for loop
    pause(.001)
end   
    cmd.position = nan(1,n);
    cmd.velocity = nan(1,n);
    cmd.torque = nan(1,n);
    group.set(cmd);