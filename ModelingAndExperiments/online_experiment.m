clc;
clear all;

%% libs
global path_to_framework path_root
path_root = "C:\Git\TesisEMG\";
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

path_to_data_for_train = horzcat(char(path_root),'Data/preprocessing/');

%% PARAMS
global userData params
global window_size stride

window_size = 300;
stride = 30;

params.window_size = window_size;
params.stride = stride;
user_full_dir = path_root +"Data/preprocessing/";
is_legacy = false;

%%
user_id_past = 308;
user_id_present = 309;

user_folder_past = "user"+user_id_past;
user_folder_present = "user"+user_id_present;

userData_past = loadUserByNameAndDir(user_folder_past, char(user_full_dir), is_legacy);
userData_present = loadUserByNameAndDir(user_folder_present, char(user_full_dir), is_legacy);

orientation_past = getOrientation(userData_past, user_folder_past);
orientation_present = getOrientation(userData_present, user_folder_past);

model_past = load(path_root + "ModelingAndExperiments/models/variation_complete/best-user" + user_id_past + ".mat");
model_present = load(path_root + "ModelingAndExperiments/models/variation_complete/best-user" + user_id_present + ".mat");

model_past_incomplete = load(path_root + "ModelingAndExperiments/models/variation_complete/incomplete-user" + user_id_past + ".mat");
model_present_incomplete = load(path_root + "ModelingAndExperiments/models/variation_complete/incomplete-user" + user_id_present + ".mat");

context_past = generateContext(params, model_past.classes_num_to_name);
context_present = generateContext(params, model_present.classes_num_to_name);

context_past_incomplete = generateContext(params, model_past_incomplete.classes_num_to_name);
context_present_incomplete = generateContext(params, model_present_incomplete.classes_num_to_name);

%%
global qnn_online context_online 
global known_dataset_training known_dataset_test
global stored_qnn stored_extra_classes
stored_qnn = struct("check",0);
stored_extra_classes = ["pinch", "open"];
model_incomplete_copy = load(path_root + "ModelingAndExperiments/models/variation_complete/incomplete-user" + user_id_past + ".mat");
transferNetwork(null(1), model_incomplete_copy, stored_extra_classes, userData_past)

global counter
global orientation
counter = 0;

orientation = getOrientation(userData_past, user_folder_past);

%%

global extra_dataset index_extra_dataset
extra_dataset = {};
index_extra_dataset = 1;
global last_sample last_prediction real_gestureName

graph_classification = [];
graph_recognition = [];

list = randperm(numel(userData_present.training));

for i=1:length(list)
    idx = list(i);
    last_sample = userData_present.training{idx};
    real_gestureName = last_sample.gestureName;
    % fprintf("Using %s->", real_gestureName);
    
    isTraining = true;
    epsilon = 0.2;
     
    context_past('user_gestures_testing') = {last_sample};
    context_past_incomplete('user_gestures_testing') = {last_sample};
    context_present('user_gestures_testing') = {last_sample};
    context_online('user_gestures_testing') = {last_sample};
    
    h_past = model_past.model.runEpisodes(@getRewardEMG, 3, context_past, 0);
    h_past_incomplete = model_past_incomplete.model.runEpisodes(@getRewardEMG, 3, context_past_incomplete, 0);
    h_present = model_present.model.runEpisodes(@getRewardEMG, 3, context_present, 0);
    h_online = qnn_online.runEpisodes(@getRewardEMG, 3, context_online, 0);
    
    
    global last_prediction_complete last_prediction
    prediction_present = string(h_present.history_responses{1}.class);
    
    last_prediction_complete = prediction_present;
    
    prediction_past = string(h_past.history_responses{1}.class);
    prediction_incomplete = string(h_past_incomplete.history_responses{1}.class);
    
    prediction_online = string(h_online.history_responses{1}.class);
    
    
    if isTraining
        class_num_to_name = context_online("classes_num_to_name");
        class_name_to_num = context_online("classes_name_to_num");
        
        Qval = zeros([1, class_num_to_name.Count]);
        Qval(class_name_to_num(prediction_online)) = 1;
       
        [~, action_index] = QLearning.selectActionQEpsilonGreedy(Qval, epsilon, length(Qval), false);
        
        prediction_online = string(class_num_to_name(action_index));
        
        
    end
    
    last_prediction = prediction_online;
    
    
    
    if string(last_sample.gestureName) == last_prediction
        giveFeedback(null(1), 1);
    else
        giveFeedback(null(1), -1);
    end
    
    [classification, recognition] = onlineQNNTest(null(1), null(1));


    graph_classification = [graph_classification, classification];
    graph_recognition = [graph_recognition, recognition];

    
    fprintf("%s - %s - %s | %s | %.2f, %.2f \n", ...
        prediction_past, prediction_incomplete, prediction_online, prediction_present, ...
        classification, recognition);

end

save(path_root+"ModelingAndExperiments/transformations/results/userVariationGraph.mat", "graph_classification", "graph_recognition");
