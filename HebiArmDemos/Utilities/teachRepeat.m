function [ endeff ] = teachRepeat( TrayArm, HebiKinematics )
%TEACHREPEAT Teach and repeat function for the TrayArm

joy = vrjoystick(1);
i = 1;
n = TrayArm.getNumModules();
sprintf('Press button 10 on Joystick to start Teach and Repeat')

cmd = CommandStruct();
cmd.position = nan(1,n);
cmd.velocity = nan(1,n);
TrayArm.set(cmd);
angles = cmd.position;
gravity = [0 0 1];

gain = GainStruct();
gain.torqueKp = ones(1,n)*.05;

TrayArm.set('gains',gain);
in = 'b';

while(~button(joy,10))
end
sprintf('Start Teach and Repeat')
while(~button(joy,9))
    angles(i,:) = TrayArm.getNextFeedback.position;
    cmd.torque = HebiKinematics.getGravCompTorques(angles(i,:),gravity);
    TrayArm.set(cmd);
    temp = HebiKinematics.getForwardKinematics('EndEffector',angles(i,:));
    in = input('HI', 's');
    if in == 'a'
        temp(1:3,4)
    end
    i = i +1;
end
% gain.torqueKp = ones(1,n)*0;
% TrayArm.set('gains',gain);
% for j = 1:length(angles(:,1))
% 
%     cmd.position = angles(j,:);
%     TrayArm.set(cmd);
%     pause(0.007);
% end

end

