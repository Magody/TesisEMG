clc;
clear all;
close all;

global path_root;
path_root = "/home/magody/programming/MATLAB/tesis/";

addpath(genpath(path_root + "GeneralLib"));
addpath(genpath(path_root + "App/matlab/lib/Myo"));
addpath(path_root + "App/matlab/lib/TimersHandle");
addpath(genpath(path_root + "App/matlab/utils/timers"));
addpath(path_root + "App/matlab/models/device");
addpath(genpath(path_root + "App/matlab/data"));
addpath(path_root + "App/matlab/media/UI");
addpath(genpath("/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework"));


%% Parameters
global deviceType myoObject
deviceType = DeviceName.myo;


global window_size stride orientation sample_time_ms
window_size = 300;
stride = 30;
user_id = 1;
user_full_dir = path_root +"Data/preprocessingTest/";
user_folder = "user"+user_id;
is_legacy = false;
userData = loadUserByNameAndDir(user_folder, char(user_full_dir), is_legacy);
rng('default');
orientation = getOrientation(userData, user_folder);

sample_time_ms = 996;

global model context;
addpath(genpath(path_root + "ModelingAndExperiments/RLSetup"));
context = containers.Map();
context('tabulation_mode') = 2;
context('is_preprocessed') = true;
context('noGestureDetection') = false;
context('window_size') = window_size;
context('stride') = stride;
context('rewards') = struct('correct', 1, 'incorrect', -1);
context('RepTesting') = 1;
context('rangeDownTest') = 1;
    
model = load(path_root + "ModelingAndExperiments/models_output/" + user_folder + ".mat");



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

%% Read sensor and get features

myoObject.myoData.setSimulationGestures("pinch");

number_gestures_to_simulate = 2;
number_integer = floor(number_gestures_to_simulate);
excedent = number_gestures_to_simulate - number_integer;
if number_integer >= 1
    if number_integer == 1
        wait_simulation = 0.9;
    else
        wait_simulation = 0.9 + ((0.1 + 0.9) * number_integer - 1);
    end
else
    wait_simulation = -0.2* excedent;
end
wait_simulation = wait_simulation +  (0.1 + 0.9) * excedent;

if deviceType == DeviceName.myo
    myoObject.myoData.stopStreaming();
    myoObject.myoData.clearLogs();
    myoObject.myoData.startStreaming();
    % start executes inmediately
end
pause(wait_simulation);
myoObject.myoData.stopStreaming();

emg_stored_length = size(myoObject.myoData.emg_log, 1);
sample_begin = max(1, emg_stored_length-sample_time_ms+1);
emg = myoObject.myoData.emg_log(sample_begin:emg_stored_length, :);

tic_toc_feature_extraction = tic;
features_per_window = extractFeaturesByWindowStride(path_root, orientation, window_size, stride, emg);
time_used_for_feature_extraction_ms = toc(tic_toc_feature_extraction) * 1000;
fprintf("40 features obtained in %d windows, collected %d\n", ...
        size(features_per_window, 1), emg_stored_length);
    
features_name = "features_per_windowWin" + window_size + "Stride" + stride;
    
sample = struct("emg", emg, features_name, features_per_window);
context('user_gestures_test') = {sample};

tic_toc_qnn_prediction = tic;
history_test = model.q_neural_network.runEpisodes(@getRewardEMG, 3, context, 0);  
time_used_for_qnn_prediction_ms = toc(tic_toc_qnn_prediction) * 1000 ;

time_used_total_ms = time_used_for_feature_extraction_ms + time_used_for_qnn_prediction_ms;
    
fprintf("*****\nPrediction %s in %.2f[ms]. \n->%.2f[ms] (%.1f %%) used in feature extraction and \n->%.2f[ms] (%.1f %%) used in qnn prediction\n", ...
        string(history_test.history_responses{1}.class), ...
        time_used_total_ms, ...
        time_used_for_feature_extraction_ms, ...
        100*time_used_for_feature_extraction_ms/time_used_total_ms, ...
        time_used_for_qnn_prediction_ms, ...
        100*time_used_for_qnn_prediction_ms/time_used_total_ms)

plot(emg);

%% Test timer to get features

timer_snapshotEMG = timer('Name', 'timer_snapshotEMG', 'TimerFcn', @timerSnapshotEMG, ...
          'ExecutionMode', 'fixedRate', 'Period', 1);
start(timer_snapshotEMG);

if deviceType == DeviceName.myo
    myoObject.myoData.stopStreaming();
    myoObject.myoData.clearLogs();
    myoObject.myoData.startStreaming();
    drawnow();
end
pause(3);
stop(timer_snapshotEMG);
myoObject.myoData.stopStreaming();


%% Read other variables of sensor

% Clean up: reset all
if deviceType == DeviceName.myo
    myoObject.myoData.stopStreaming();
    myoObject.myoData.clearLogs();
    myoObject.myoData.startStreaming();
    drawnow();
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


%% plot figure in real time
myoObject.myoData.setSimulationGestures("fist");
if deviceType == DeviceName.myo
    myoObject.myoData.stopStreaming();
    myoObject.myoData.clearLogs();
    myoObject.myoData.startStreaming();
    drawnow();
end

global h plot_index_begin;
h = animatedline(0,0,'Color','b');
plot_index_begin = 1;

timer_plotAnimatedEMG = timer('Name', 'timer_plotAnimatedEMG', 'TimerFcn', @timerPlotAnimatedEMG, ...
          'ExecutionMode', 'fixedRate', 'Period', 0.2);
start(timer_plotAnimatedEMG);

pause(1);
myoObject.myoData.stopStreaming();
pause(0.2); % plot the very last part collected
stop(timer_plotAnimatedEMG);

%% Disconnect
% We can stop streaming
% myoObject.myoData.stopStreaming()

% Or disconnect full, erasing some other variables
disconnectMyo(myoObject);
cleanAllTimers(); % sanity check

%% dump

a = animatedline;
for i=1:5
    index_begin = (i-1)*10;
    index_end = index_begin + 10;
    addpoints(a, (index_begin+1):index_end, rand([1, 10]));
    drawnow
end