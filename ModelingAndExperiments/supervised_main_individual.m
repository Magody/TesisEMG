%% Lib and dirs
clc;
clear all;
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";

path_root = "/home/magody/programming/MATLAB/tesis/";

addpath(genpath(path_to_framework));
addpath(genpath(path_root + "ModelingAndExperiments/utils"));
addpath(path_root + "ModelingAndExperiments/learning")
addpath(path_root + "ModelingAndExperiments/Experiments")
addpath(genpath(path_root + "GeneralLib"));

path_output = path_root + "ModelingAndExperiments/models/models_supervised/";

path_to_data_for_train = horzcat(char(path_root),'Data/preprocessingTest/'); % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';
path_to_data_for_testing = horzcat(char(path_root),'Data/preprocessingTest/'); % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';

jsonName=horzcat(char(path_output), 'responses.json');
version = 'testing';

%% set parameters
verbose_level = 1;
experiment_id = 10;  % 10 11
experiment_mode = "individual";

experiments_csv = readtable(path_root + 'ModelingAndExperiments/Experiments/experiments_parameters_QNN.csv');
params_experiment_row = experiments_csv(experiment_id, :);
        
[params, hyperparams] = build_params(params_experiment_row, experiment_mode, verbose_level);

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

do_test = false;
% This script is for individual model only
for user_id=1:num_users
    try
        user_folder = "user"+user_id;
        params.model_dir_name = path_output + params.model_name + "-" + user_folder + ".mat";    
        [context('user_gestures_training'), context('user_gestures_validation'), ...
            dataset_testing] = ...
            splitUserDataIndividual(user_folder, path_to_data_for_train, ignoreGestures, params.porc_training, params.porc_validation, "packet");
        
        [X_train, y_train] = mergeAndGetXy(context('user_gestures_training'), params.window_size, params.stride, context('classes_name_to_num'));
        
        
        do_validation = params.porc_validation > 0;
        
        if do_validation
            [X_validation, y_validation] = mergeAndGetXy(context('user_gestures_validation'), params.window_size, params.stride, context('classes_name_to_num'));
        else
            X_validation = [];
            y_validation = [];
        end
        neural_network = buildNeuralNetwork(hyperparams);

        history_train_validation = neural_network.train(X_train, y_train, X_validation, y_validation, params.verbose_level-1);
        
        history_errors = history_train_validation('history_errors');
        history_accuracy_validation = history_train_validation('history_accuracy_validation');
        
        history_episodes = struct("history_errors", history_errors);
        summary = struct();
        if do_test
            history_episodes = evaluateRecognitionSupervised(neural_network, 2, dataset_testing, context, params.verbose_level-1);

            [classification_window_test, classification_test, recognition_test] = Experiment.getEpisodesEMGMetrics(history_episodes);

            summary = struct("classification_window_test", classification_window_train, ...
                                       "classification_test", classification_train, ...
                                       "recognition_test", recognition_train);


            %{
            figure(1);
            subplot(1,2,1);
            plot(history_errors);
            subplot(1,2,2);
            plot(history_accuracy_validation);
            %}

            accuracy_classification_window = accuracy_classification_window + summary.classification_window_test.accuracy;
            accuracy_classification = accuracy_classification + summary.classification_test.accuracy;
            accuracy_recognition = accuracy_recognition + summary.recognition_test.accuracy;

        
        end

        ExperimentHelper.saveModel(params.model_dir_name, neural_network, history_episodes, summary, context);

    catch exception
        fprintf("Error with %s, \n%s\n", user_folder, exception.message + "->" + ...
                exception.stack(1).name + " line: " + exception.stack(1).line);
    end
end
t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
if params.verbose_level > 0 && do_test
    fprintf("Final class: %.4f, Final recognition: %.4f\n", ...
        accuracy_classification/num_users, accuracy_recognition/num_users); 
end

%% Test

run_test_as_validation = false;
use_same_training_set_as_validation = true;

accuracy_classification = 0;
accuracy_recognition = 0;

Rep = 150;  % below it is not controlled, is with length(responses)

results = struct();

type_execution = 3;
if run_test_as_validation
    type_execution = 2;
end

for i=1:num_users
    try
        user_folder = users(i).name;
        
        context('offset_user') = 0;
        
        params.model_dir_name = path_output + params.model_name + "-" + user_folder + ".mat";    
        model = load(params.model_dir_name);
        
        neural_network = model.model;
        
        [context('user_gestures_training'), context('user_gestures_validation'), dataset_testing] = ...
            splitUserDataIndividual(user_folder, path_to_data_for_testing, ignoreGestures, params.porc_training, params.porc_validation, "normal");
        
        % Test
        if run_test_as_validation
            if use_same_training_set_as_validation
                dataset_testing = context('user_gestures_training');
            else
                dataset_testing = context('user_gestures_validation');
            end
        end
        history_episodes = evaluateRecognitionSupervised(neural_network, type_execution, dataset_testing, context, params.verbose_level-1);
        
        responses = history_episodes.history_responses;
        [classification_window_test, classification_test, recognition_test] = Experiment.getEpisodesEMGMetrics(history_episodes);
        
        summary = struct("classification_window_test", classification_window_test, ...
                                   "classification_test", classification_test, ...
                                   "recognition_test", recognition_test);
                               
        
        if run_test_as_validation
            accuracy_classification = accuracy_classification + summary.classification_test.accuracy;
            accuracy_recognition = accuracy_recognition + summary.recognition_test.accuracy;
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
       accuracy_classification/num_users, accuracy_recognition/num_users); 
end

%% Export  in json
jsonFormat(jsonName, results);




