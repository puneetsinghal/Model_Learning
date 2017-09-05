function [ xyz] = getCoordinates()
%% GETCOORDINATES gets world coordinates from pixel location
% Author: Garrison Johnston	
%% Run Python Script

oscmd = 'C:\Python27\python.exe C:\Users\Johnsg7\Desktop\CMUInternship\ModelLearning\HebiArmDemos\DetectYGreenBall.py'; % change this for your system
[status,cmdout] = system(oscmd);

%% Data Type Conversion
str = strsplit(cmdout);
xy_pix = str2double(str);

%% Convert from pixels to meters
radius_m = (58.86/1000)/2;
pix2meters = radius_m/xy_pix(3);
x = (xy_pix(1) - 182.091629028)*(0.1214+0.1375)/(402.122-182.0916) - 0.1375;
y = (xy_pix(2) - 265.1924)*(0.3938-0.2573)/(165.0158-265.1924) + 0.2573;
dist = (xy_pix(3) - 24.1008)*(0.2460-0.0275)/(31.4403-24.1008) + 0.0275;
% x = pix2meters*(xy_pix(1)-600/2);
% y = pix2meters*(xy_pix(2)-375/2);
% T = [1 0 0 0.1016; ...
%      0 1 0 0.3937; ...
%      0 0 1 0; ...
%      0 0 0 1;];
% xyz = T*[x;y;dist+0.01;1];
% xyz = [x,y,dist+0.01];
if (x > 0)
    xyz = [x+0.13, y - 0.045, dist + 0.02];
else
    xyz = [x+0.025, y - 0.045, dist+0.02];
end
end
