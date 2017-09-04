function collisionStop (fbk, cmd, r)
i = 1;
eps = [5, 4.5, 5, 5, 5];
torques = fbk.torque;
xyz = fbk.position;
while i < 6
    if abs(torques(i) - cmd.torque(i)) > eps(i)
        disp('Collision Detected. Stopping...')
        i
        while(1)
            cmd.position = xyz;
            r.set(cmd);
            pause(0.01);
        end
    end
    i = i + 1;
end

