function gripperOff(a)
writeDigitalPin(a, 'D52', 0);
writePWMDutyCycle(a, 'D2', 0);
end