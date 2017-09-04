function gripperOn(a)
writeDigitalPin(a, 'D52', 1);
writePWMDutyCycle(a, 'D2', 0.5);
end