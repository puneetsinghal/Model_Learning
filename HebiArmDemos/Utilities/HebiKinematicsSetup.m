kin = HebiKinematics();
kin.addBody('X5-9', 'PosLim', [0 2*pi]);
kin.addBody('GenericLink','com',[.1 .1 .1],...
            'out',([eye(4,3),[-.022 0 .047 1]']*(rotz(pi/2)*rotx(pi/2))),...
            'mass',0.1);
kin.addBody('X5-9', 'PosLim', [-pi/2 pi/2]);
kin.addBody('GenericLink','com',[0 0 0],'out',rotz(pi/2),'mass',0);
kin.addBody('X5Link', 'ext', 0.280, 'twist', pi);
kin.addBody('X5-9', 'posLim', [-.1 .9*pi]);
kin.addBody('X5Link', 'ext', 0.280, 'twist', pi);
kin.addBody('X5-4','posLim', [-pi pi]);
kin.addBody('GenericLink','com',[.1 .1 .1],...
            'out',([eye(4,3),[0.04 0 .047 1]']*(rotz(pi/2)*rotx(pi/2))),...
            'mass',0.1);
kin.addBody('X5-1');

%print configuration information to screen (uncheck when you want to see)
kin.getBodyInfo();
kin.getJointInfo();

newFrame = eye(4)*rotz(-pi/2);
kin.setBaseFrame(newFrame);

%confirm zero-position of arm matches to physical arm
frames = kin.getForwardKinematics('output',[0 0 0 0 0]);
%%
%3. Create the HebiKinematics group
macs = {'D8:80:39:E8:B3:3C';'D8:80:39:9B:37:7F';'D8:80:39:9D:29:4E';...
        'D8:80:39:9D:3F:9D';'D8:80:39:9D:04:78'};
group = HebiLookup.newGroupFromMacs(macs);