clc;
clear all;
close all;

global path_root;
path_root = "C:/Git/TesisEMG/";
path_to_framework = "C:/Git/MATLABMagodyFramework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";

addpath(genpath(path_root + "ModelingAndExperiments/RLSetup"));
addpath(genpath(path_root + "ModelingAndExperiments/utils"));
addpath(genpath(path_root + "ModelingAndExperiments/Experiments"));
addpath(genpath(path_root + "ModelingAndExperiments/learning"));
addpath(genpath(path_root + "GeneralLib"));
addpath(genpath(path_root + "App/matlab/lib/Myo"));
addpath(path_root + "App/matlab/lib/TimersHandle");
addpath(genpath(path_root + "App/matlab/utils/timers"));
addpath(path_root + "App/matlab/models/device");
addpath(genpath(path_root + "App/matlab/data"));
addpath(genpath(path_root + "App/matlab/media"));
addpath(genpath(path_to_framework));

%% Parameters
global deviceType myoObject
deviceType = DeviceName.myo;


global window_size stride orientation sample_time_ms
window_size = 300;
stride = 30;
params.window_size = window_size;
params.stride = stride;
user_id = 1;
user_full_dir = path_root +"Data/preprocessingTest/";
user_folder = "user"+user_id;
is_legacy = false;
rng('default');

sample_time_ms = 996;

global model_complete model_incomplete context_for_model_complete context_for_model_incomplete;

userData = loadUserByNameAndDir(user_folder, char(user_full_dir), is_legacy);
orientation = getOrientation(userData, user_folder);

data_dir1 = path_root + "App/matlab/data/models_complete/complete-user" + user_id + ".mat";
data_dir2 = path_root + "App/matlab/data/models_complete/complete-user" + user_id + ".mat";

model_complete = load(data_dir1);
model_incomplete = load(data_dir2);

context_for_model_complete = generateContext(params, model_complete.classes_num_to_name);
context_for_model_incomplete = generateContext(params, model_incomplete.classes_num_to_name);

%% timers check
myWaitbarTimer = timerWaitbar(2);
start(myWaitbarTimer);

%% Clean
cleanAllTimers();

%% Start connection
disp('CONNECTING, please, wait...');

if deviceType == DeviceName.myo

    % [myoObject, isConnectedMyo] = connectMyo("fake");
    [myoObject, isConnectedMyo] = connectMyo("real");

    if isConnectedMyo
        disp('Device connected');
    else
        disp('Could not connect Myo with Matlab');
    end
else
    fprintf("Device not supported\n");
end

% Sanity check

if deviceType == DeviceName.myo
    myoObject.myoData.stopStreaming();
    myoObject.myoData.clearLogs();
    myoObject.myoData.startStreaming();
end

% myWaitbarTimer = timerWaitbar(2);
% start(myWaitbarTimer);

activate = true;
l = length(myoObject.myoData.emg_log);
last_norm = -1;
while l < 1100
    norm = floor(l/10) * 10;
    if mod(norm, 100) == 0
        if norm ~= last_norm
            last_norm = norm;
            disp(l);
        end
    end
    if activate && l > 100 && l < 200
        disp("BEGIN!!!");
        activate = false;
    end
    l = length(myoObject.myoData.emg_log);
end
disp("END!!");

myoObject.myoData.stopStreaming();
emg_sanity = myoObject.myoData.emg_log;
% Or disconnect full, erasing some other variables
disconnectMyo(myoObject);
cleanAllTimers(); % sanity stop

plot(emg_sanity)


%% Read sensor and get features

myoObject.myoData.setSimulationGestures("pinch");

number_gestures_to_simulate = 1;
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
context_for_model_complete('user_gestures_test') = {sample};
context_for_model_incomplete('user_gestures_test') = {sample};

tic_toc_qnn_prediction = tic;
history_test_complete = model_complete.q_neural_network.runEpisodes(@getRewardEMG, 3, context_for_model_complete, 0);  
history_test_incomplete = model_incomplete.q_neural_network.runEpisodes(@getRewardEMG, 3, context_for_model_incomplete, 0);  
time_used_for_qnn_prediction_ms = toc(tic_toc_qnn_prediction) * 1000 ;

time_used_total_ms = time_used_for_feature_extraction_ms + time_used_for_qnn_prediction_ms;
    
fprintf("*****\nPrediction by model [complete: %s, incomplete: %s] in %.2f[ms]. \n->%.2f[ms] (%.1f %%) used in feature extraction \n->%.2f[ms] (%.1f %%) used in qnn prediction\n", ...
        string(history_test_complete.history_responses{1}.class), ...
        string(history_test_incomplete.history_responses{1}.class), ...
        time_used_total_ms, ...
        time_used_for_feature_extraction_ms, ...
        100*time_used_for_feature_extraction_ms/time_used_total_ms, ...
        time_used_for_qnn_prediction_ms, ...
        100*time_used_for_qnn_prediction_ms/time_used_total_ms)

figure(1);
plot(emg);

%% modify architecture and transfer knowledge

classes_to_expand = 1;

output_neurons_new = prod(model_incomplete.q_neural_network.sequential_network.shape_output) + classes_to_expand;

sequential_network_length = length(model_incomplete.q_neural_network.sequential_network.network);
network_new = cell([1, sequential_network_length]);
layers_range_to_copy = 1:sequential_network_length-1;
network_new(layers_range_to_copy) = model_incomplete.q_neural_network.sequential_network.network(layers_range_to_copy);
% last layer should be new due to different neurons
network_new(sequential_network_length) = {Dense(output_neurons_new, "xavier")};


qnn_online = QNeuralNetwork(Sequential({}), Sequential(network_new), model_incomplete.q_neural_network.nnConfig, model_incomplete.q_neural_network.qLearningConfig, model_incomplete.q_neural_network.functionExecuteEpisode);
qnn_online.transferGameReplay(model_incomplete.q_neural_network.gameReplay);
qnn_online.setCustomRunEpisodes(@customRunEpisodesEMG);

classes_num_to_name = model_incomplete.classes_num_to_name;
classes_num_to_name(5) = "pinch";
context = generateContext(params, classes_num_to_name);

%% Prepare reference dataset
dataset_complete = userData.training;
known_dataset = {};
known_classes = string(values(model_incomplete.classes_num_to_name));
for index_dataset_complete=1:length(dataset_complete)
    gesture = dataset_complete{index_dataset_complete}.gestureName;
    if searchStringInArray(known_classes, gesture) ~= -1
        known_dataset = [known_dataset, dataset_complete(index_dataset_complete)];
    end
end

%% Train qnn_transfer

% Filter configuration
options.rectFcn = 'abs'; % Function for emg rectification
[options.Fb, options.Fa] = butter(4, 0.05, 'low'); % Filter values
options.detectMuscleActivity = true; % Activates the detection of muscle activity
options.fs = 200; % Sampling frequency of the emg
options.minWindowLengthOfMuscleActivity = 150;
options.threshForSumAlongFreqInSpec = 12; % Threshold for detecting the muscle activity
options.minWindowLengthOfSegmentation = 150; % PENDIENTE
options.plotSignals = false;


groundTruthIndex = [0, 0];
[groundTruthIndex(1), groundTruthIndex(2)] = detectMuscleActivity(emg, options);

plot(emg)
rectangle('Position',[groundTruthIndex(1) -1  groundTruthIndex(2)-groundTruthIndex(1) 2],'EdgeColor','g')


gesture_mistake = struct("emg", emg, features_name, features_per_window, ...
                         "gestureName", "pinch", "groundTruthIndex", groundTruthIndex);
for i=1:25
    % data augmentation of the mistake
    known_dataset = [known_dataset, {gesture_mistake}];
end

known_dataset = known_dataset(randperm(numel(known_dataset)));

do_validation = true;
context('user_gestures_train') = known_dataset(1:100);
if do_validation
    context('user_gestures_validation') = known_dataset(101:125);
end

[history_episodes_by_epoch, summary, ~] = trainAndValidate(path_to_framework, path_root, ...
                                            qnn_online, 3, ...
                                            do_validation, context, 1);
%% Test again


context('user_gestures_test') = {gesture_mistake};
context_for_model_complete('user_gestures_test') = {gesture_mistake};
context_for_model_incomplete('user_gestures_test') = {gesture_mistake};
history_test_complete = model_complete.q_neural_network.runEpisodes(@getRewardEMG, 3, context_for_model_complete, 0);  
history_test_incomplete = model_incomplete.q_neural_network.runEpisodes(@getRewardEMG, 3, context_for_model_incomplete, 0);  
history_test_transfer = qnn_online.runEpisodes(@getRewardEMG, 3, context, 0);  

fprintf("Prediction by [correct=%s, incorrect=%s, transfer=%s]\n", ...
        string(history_test_complete.history_responses{1}.class), ...
        string(history_test_incomplete.history_responses{1}.class), ...
        string(history_test_transfer.history_responses{1}.class));

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