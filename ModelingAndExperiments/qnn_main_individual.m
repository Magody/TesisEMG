%% Lib and dirs
clc;
clear all;
path_root = "C:/Git/TesisEMG/";
path_to_framework = "C:/Git/MATLABMagodyFramework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";

addpath(genpath(path_to_framework));
addpath(path_root + "ModelingAndExperiments/utils")
addpath(path_root + "ModelingAndExperiments/learning")
addpath(path_root + "ModelingAndExperiments/RLSetup")
addpath(path_root + "ModelingAndExperiments/Experiments")
addpath(genpath(path_root + "GeneralLib"));

path_output = path_root + "ModelingAndExperiments/models/models_unique/";  % test_low_umbral

path_to_data_for_train = horzcat(char(path_root),'Data/preprocessing/'); % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';
path_to_data_for_testing = horzcat(char(path_root),'Data/preprocessing/'); % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';

jsonName=horzcat(char(path_output), 'responses.json');
version = 'testing';

%% set parameters
verbose_level = 2;
experiment_id = 1;
experiment_mode = "individual";


experiments_csv = readtable(path_root + 'ModelingAndExperiments/Experiments/experiments_parameters_QNN.csv');
params_experiment_row = experiments_csv(experiment_id, :);
        
[params, hyperparams] = build_params(params_experiment_row, experiment_mode, verbose_level);
hyperparams.executeEpisodeEMG = @executeEpisodeEMG;
hyperparams.customRunEpisodesEMG = @customRunEpisodesEMG;

gestures_list = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"];
ignoreGestures = [];
classes_num_to_name = getClassNumToName(gestures_list, ignoreGestures);    
context = generateContext(params, classes_num_to_name);

users = dir(path_to_data_for_testing);
users = users(3:end);
num_users = length(users);

%% Train and validate
t_begin = tic;
accuracy_classification_window = 0;
accuracy_classification = 0;
accuracy_recognition = 0;
% This script is for individual model only
for user_id=307:307 % num_users
    try
        user_folder = "user"+user_id;
        params.qnn_model_dir_name = path_output + params.model_name + "-" + user_folder + ".mat";    
        [context('user_gestures_training'), context('user_gestures_validation'), context('user_gestures_testing')] = ...
            splitUserDataIndividual(user_folder, path_to_data_for_train, ignoreGestures, params.porc_training, params.porc_validation, "packet");
        total_episodes = hyperparams.general_epochs * length(context('user_gestures_training')); % if is general: * num_users;
        q_neural_network = buildQNeuralNetwork(hyperparams, total_episodes);

        do_validation = params.porc_validation > 0;

        [history_episodes_by_epoch, summary, ~] = trainAndValidate(path_to_framework, path_root, q_neural_network, ...
                                            hyperparams.general_epochs, do_validation, context, params.verbose_level-1);

        if do_validation
            summary{hyperparams.general_epochs, 2}
            accuracy_classification_window = accuracy_classification_window + summary{hyperparams.general_epochs, 2}.classification_window_validation.accuracy;
            accuracy_classification = accuracy_classification + summary{hyperparams.general_epochs, 2}.classification_validation.accuracy;
            accuracy_recognition = accuracy_recognition + summary{hyperparams.general_epochs, 2}.recognition_validation.accuracy;
        else
            accuracy_classification_window = accuracy_classification_window + summary{hyperparams.general_epochs, 1}.classification_window_train.accuracy;
            accuracy_classification = accuracy_classification + summary{hyperparams.general_epochs, 1}.classification_train.accuracy;
            accuracy_recognition = accuracy_recognition + summary{hyperparams.general_epochs, 1}.recognition_train.accuracy;
        end

        ExperimentHelper.saveModel(params.qnn_model_dir_name, q_neural_network, history_episodes_by_epoch, summary, context);

    catch exception
        fprintf("Error with %s, \n%s\n", user_folder, exception.message + "->" + ...
                exception.stack(1).name + " line: " + exception.stack(1).line);
    end
end
t_end = toc(t_begin);
if params.verbose_level > 0
    fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
    fprintf("Final class: %.4f, Final recognition: %.4f\n", ...
        accuracy_classification/num_users, accuracy_recognition/num_users); 
end

%% Test

run_test_as_validation = true;
use_same_training_set_as_validation = true;

accuracy_classification = 0;
accuracy_recognition = 0;

Rep = 150;  % below it is not controlled, is with length(responses)

results = struct();

for i=1:num_users  % 3 for user100
    try
        user_folder = users(i).name;
        
        if user_folder == "user115" || user_folder == "user209"
            continue;
        end
        
        if user_folder ~= "user10"
           continue; 
        end
        
        context('offset_user') = 0;
        
        % params.model_name + "-" + 
        params.qnn_model_dir_name = path_output + params.model_name + "-" + user_folder + ".mat";    
        qnn_model = load(params.qnn_model_dir_name);
        
        q_neural_network = qnn_model.model; % q_neural_network;
        
        [context('user_gestures_training'), context('user_gestures_validation'), context('user_gestures_testing')] = ...
            splitUserDataIndividual(user_folder, path_to_data_for_testing, ignoreGestures, params.porc_training, params.porc_validation, "normal");
        
        % Test
        if run_test_as_validation
            if use_same_training_set_as_validation
                context('user_gestures_validation') = context('user_gestures_training');
            end
        end
        [summary, responses, history_test] = ExperimentHelper.testModelIndividual(q_neural_network, run_test_as_validation, context, verbose_level-1);
        
        if run_test_as_validation
            accuracy_classification = accuracy_classification + summary.classification_test;
            accuracy_recognition = accuracy_recognition + summary.recognition_test;
        end
        
        for j =  0:length(responses)-1
            sample = sprintf('idx_%d',j);

            results.(version).(user_folder).class.(sample)                  = responses{j+1}.class ;
            results.(version).(user_folder).vectorOfLabels.(sample)         = responses{j+1}.vectorOfLabels;
            results.(version).(user_folder).vectorOfTimePoints.(sample)     = responses{j+1}.vectorOfTimePoints;
            results.(version).(user_folder).vectorOfProcessingTime.(sample) = responses{j+1}.vectorOfProcessingTimes;

        end
        
    catch exception
        fprintf("Test: Error with %s, \n%s\n", user_folder, exception.message + "->" + ...
                exception.stack(1).name + " line: " + exception.stack(1).line);
    end
end


if run_test_as_validation
   fprintf("Mean final class: %.4f, Final recognition: %.4f\n", ...
       accuracy_classification/(num_users-2), accuracy_recognition/(num_users-2)); 
end

%% Export  in json
jsonFormat(jsonName, version, Rep, results);