clc;
clear all;
close all;
addpath(genpath('lib/Myo'));
addpath('lib/TimersHandle');
addpath('utils/plot');
addpath('models/device');
addpath('media/UI');

%% Parameters
global deviceType myoObject
deviceType = DeviceName.myo;

%% Start connection
disp('CONNECTING, please, wait...');

if deviceType == DeviceName.myo

    [myoObject, isConnectedMyo] = connectMyo("fake");
    % [myoObject, isConnectedMyo] = connectMyo();

    if isConnectedMyo
        disp('Device connected');
    else
        disp('Could not connect Myo with Matlab');
    end
else
    fprintf("Device not supported\n");
end



%%

% Clean up: reset all
if deviceType == DeviceName.myo
    myoObject.myoData.stopStreaming();
    myoObject.myoData.clearLogs();
    myoObject.myoData.startStreaming();
end

%{
Get data
sample = struct('emg', [], 'quaternions', [], 'gyro', [],'accel', [], ...
    'gestureDevicePredicted', []);


if deviceType == DeviceName.myo
    
    sample.emg = myoObject.myoData.emg_log;
    sample.gestureDevicePredicted = myoObject.myoData.pose_log;
    
    if isempty(sample.emg)
        errorInData = true;
        errorType = 'noData';
    end

    if any(sample.gestureDevicePredicted == 65535)
        errorInData = true;
        errorType = '65535';
    end

    sample.quaternions = myoObject.myoData.quat_log();% w,x,y,z
    sample.gyro = myoObject.myoData.gyro_log();
    sample.accel =  myoObject.myoData.accel_log();
        
else
    % not supported yet
    fprintf("Device not supported\n");
end
%}


%%
global h plot_call_counter;
h = animatedline;
plot_call_counter = 0;

timer_plotAnimatedEMG = timer('Name', 'timer_plotAnimatedEMG', 'TimerFcn', @plotAnimatedEMG, ...
          'ExecutionMode', 'fixedRate', 'Period', 0.2);
start(timer_plotAnimatedEMG);

%% Disconnect
% We can stop streaming
% myoObject.myoData.stopStreaming()

% Or disconnect full, erasing some other variables
disconnectMyo(myoObject);
cleanAllTimers(); % sanity check