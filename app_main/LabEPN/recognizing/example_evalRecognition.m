
% libs
addpath(genpath(pwd))

%% 
% GROUND TRUTH (no depende del modelo)
repInfo.gestureName = categorical({'waveIn'});
repInfo.groundTruth = false(1, 1000);
repInfo.groundTruth(200:400) = true;

plot(repInfo.groundTruth)


%% PREDICCION
response.vectorOfLabels = categorical({'noGesture', 'noGesture', 'waveIn', 'noGesture', 'noGesture', 'noGesture'});
response.vectorOfTimePoints = [40 200 400 600 800 999]; %1xw double

% con un tic toc.
% tiempo de procesamiento 
response.vectorOfProcessingTimes = [0.1 0.1 0.1 0.1 0.1 0.1]; % 1xw double

% no necesariamente depende del vector de arriba
response.class = categorical({'waveIn'}); %eval classif

%%
r1 = evalRecognition(repInfo, response)
