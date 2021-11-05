function groundTruthIndex = getMuscularActivityGroundTruth(emg)
    % Filter configuration
    options.rectFcn = 'abs'; % Function for emg rectification
    [options.Fb, options.Fa] = butter(4, 0.05, 'low'); % Filter values
    options.detectMuscleActivity = true; % Activates the detection of muscle activity
    options.fs = 100; % Sampling frequency of the emg
    options.minWindowLengthOfMuscleActivity = 150;
    options.threshForSumAlongFreqInSpec = 15; % Threshold for detecting the muscle activity
    options.minWindowLengthOfSegmentation = 150; % PENDIENTE
    options.plotSignals = false;
    groundTruthIndex = [0, 0];
    [groundTruthIndex(1), groundTruthIndex(2)] = detectMuscleActivity(emg, options);
end

