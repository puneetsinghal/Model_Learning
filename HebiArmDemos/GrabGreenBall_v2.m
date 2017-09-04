%%This script makes 5dof arm grab green ball
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
%1. 8-10-2017        Garrison Johnston     Creation file
%
%-------------------------------------------------------------------------
%%
%clear plots and workspace
close all
clc
clear all
%%
%1. Add the path of the Hebi SDK and Plotting tools to the workspace
addpath(genpath('C:\Users\Johnsg7\Desktop\CMUInternship\Model Learning\Computer Vision\Matlab demos\Hebi Arm Demos\hebi'));
addpath(genpath('C:\Users\Johnsg7\Desktop\CMUInternship\Model Learning\Computer Vision\Matlab demos\Hebi Arm Demos\matlab_SEA'));
addpath(genpath('C:\Users\Johnsg7\Desktop\CMUInternship\Model Learning\Computer Vision\Matlab demos\Hebi Arm Demos\Utilities'));
%%
%2. Setup the kinematics configuration of the 5-DOF arm
HebiKinematicsSetup;
n = group.getNumModules;
a = arduino();
%%
%3. set gains and limits
%setGains;

%4. create command structure
cmd = CommandStruct();

%%
%initially disable all motors
cmd.position = nan(1,n);
cmd.velocity = nan(1,n);
cmd.torque = nan(1,n);
group.set(cmd);
pause(1);
gravity = [0 0 1];
ball_count = 1;
%% main loop
while(1)
    %% get ball position
    ballCoordinates = getCoordinates();
    bc = [ballCoordinates(1), ballCoordinates(2), ballCoordinates(3)];
    fbk = group.getNextFeedback();
    endeffector = kin.getForwardKinematics('EndEffector',fbk.position);
    xyz = endeffector(1:3,4);
    %% Command Arm
    
    trajz = linspace(xyz(3),  bc(3));
    for i = 1:1:length(trajz)
        fbk = group.getNextFeedback();
        positions = kin.getInverseKinematics('xyz',[xyz(1), xyz(2), trajz(i) + 0.01],'TipAxis',...
            [0 0 -1], 'InitialPositions',fbk.position,'MaxIterations',10000);
        %set gravity compensation
        cmd.torque = kin.getGravCompTorques(fbk.position,gravity);
        cmd.position = positions;
%         collisionStop(fbk,cmd, group);
        group.set(cmd);
        pause(.01);
    end
    
    fbk = group.getNextFeedback();
    endeffector = kin.getForwardKinematics('EndEffector',fbk.position);
    xyz = endeffector(1:3,4);
    trajx = linspace(xyz(1), bc(1));
    trajy = linspace(xyz(2), bc(2), length(trajx));
    for i = 1:1:length(trajx)
        fbk = group.getNextFeedback();
        positions = kin.getInverseKinematics('xyz',[trajx(i), trajy(i), bc(3) + 0.03],'TipAxis',...
            [0 0 -1], 'InitialPositions',fbk.position,'MaxIterations',10000);
        %set gravity compensation
        
        cmd.torque = kin.getGravCompTorques(fbk.position,gravity);
        cmd.position = positions;
%         collisionStop(fbk,cmd, group);
        group.set(cmd);
        pause(.01);
        torques = fbk.torque;
    end
    
    fbk = group.getNextFeedback();
    endeffector = kin.getForwardKinematics('EndEffector',fbk.position);
    xyz = endeffector(1:3,4);
    trajz = linspace(xyz(3),  bc(3) - 0.05);
    for i = 1:1:length(trajz)
        fbk = group.getNextFeedback();
        positions = kin.getInverseKinematics('xyz',[xyz(1), xyz(2), trajz(i)],'TipAxis',...
            [0 0 -1], 'InitialPositions',fbk.position,'MaxIterations',10000);
        %set gravity compensation
        cmd.torque = kin.getGravCompTorques(fbk.position,gravity);
        cmd.position = positions;
%         collisionStop(fbk,cmd, group);
        group.set(cmd);
        pause(.01);
    end
    gripperOn(a);
    fbk = group.getNextFeedback();
    endeffector = kin.getForwardKinematics('EndEffector',fbk.position);
    xyz = endeffector(1:3,4);
    trajx = linspace(xyz(1),  -0.3);
    trajy = linspace(xyz(2),  0);
    trajz = linspace(xyz(3),  0.2);
    for i = 1:1:length(trajz)
        fbk = group.getNextFeedback();
        positions = kin.getInverseKinematics('xyz',[trajx(i), trajy(i), trajz(i)],'TipAxis',...
            [0 0 -1], 'InitialPositions',fbk.position,'MaxIterations',10000);
        %set gravity compensation
        cmd.torque = kin.getGravCompTorques(fbk.position,gravity);
        cmd.position = positions;
%         collisionStop(fbk,cmd,group);
        group.set(cmd);
        pause(.01);
    end
    i = 0;
    while(1)
        if i == 100
            gripperOff(a);
            HEBI_Arm_Joystick_control_Demo_V2;
            break
        end
        cmd.torque = kin.getGravCompTorques(fbk.position,gravity);
        group.set(cmd);
        pause(0.01);
        i = i + 1;
    end
    
end
