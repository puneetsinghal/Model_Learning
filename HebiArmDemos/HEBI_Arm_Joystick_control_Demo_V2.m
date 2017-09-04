%%This Script Run a Demo of the 5-DOF arm in the biorobotics lab capable of
%This particular demo controls the arm using a standard Joystick
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
%2. 12-Jun-2017         Puneet Singhal      Changed the up-down motion. 
%                   Right stick up command takes the robot up and right stick down command
%                   takes the robot down.
%
%-------------------------------------------------------------------------
%%
%%
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

%group.setFeedbackFrequency(100);
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
     if buttons(2) == 1
         gripperOn(a);
     elseif buttons(3) == 1
         gripperOff(a);
     end
     xyz_prev = xyz;  %save previous value of position
     xyz = xyz + [-axes(1) axes(2)  -axes(4)]'*0.002;
   
    %set gravity compensation and position commands
    gravity = [0 0 1];
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

