%%This Script Run a Teach repeat Demo
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
%1. -------         Ky Woodard         Creation file
%2. 1-Mar-2016      Kaveh Nikou        Update code for demo
%
%-------------------------------------------------------------------------
%%
%clear plots and workspace
close all
clc
clear all
%%
%1. Add the path of the Hebi SDK and Plotting tools to the workspace
addpath(genpath('C:\Users\Johnsg7\Desktop\CMUInternship\ModelLearning\ComputerVision\MatlabDemos\Hebi Arm Demos\hebi'));
addpath(genpath('C:\Users\Johnsg7\Desktop\CMUInternship\ModelLearning\ComputerVision\MatlabDemos\Hebi Arm Demos\matlab_SEA'));
addpath('C:\Users\Johnsg7\Desktop\CMUInternship\ModelLearning\ComputerVision\MatlabDemos\Hebi Arm Demos\Utilities');


HebiKinematicsSetup;
endeff = teachRepeat(group,kin);
