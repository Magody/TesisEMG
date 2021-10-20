%% Lib and dirs
clc;
clear all;
close all;
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";

path_root = "/home/magody/programming/MATLAB/tesis/";

addpath(genpath(path_to_framework));
addpath(genpath(path_root + "ModelingAndExperiments"))
addpath(genpath(path_root + "GeneralLib"));

path_to_data = horzcat(char(path_root), 'Data/preprocessing/'); 
path_output = path_root + "ModelingAndExperiments/models/models_debug/";

%% Parameters and Hyperparameters setting
verbose_level = 2;
experiment_id = 9;
experiment_mode = "individual";
run_test_as_validation = true;
use_same_training_set_as_validation = true;


experiments_csv = readtable(path_root + 'ModelingAndExperiments/Experiments/experiments_parameters_QNN.csv');
params_experiment_row = experiments_csv(experiment_id, :);
        
[params, hyperparams] = build_params(params_experiment_row, experiment_mode, verbose_level);
hyperparams.executeEpisodeEMG = @executeEpisodeEMG;
hyperparams.customRunEpisodesEMG = @customRunEpisodesEMG;

gestures_list = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"];
ignoreGestures = [];
classes_num_to_name = getClassNumToName(gestures_list, ignoreGestures);    
context = generateContext(params, classes_num_to_name);

%% Train QNN
t_begin = tic;
num_users = length(params.list_users);
accuracy_classification_window = 0;
accuracy_classification = 0;
accuracy_recognition = 0;
% This script is for individual model only
for user_id=208:208 % num_users
    try
        user_folder = "user"+user_id;
        params.qnn_model_dir_name = path_output + params.model_name + "-" + user_folder + ".mat";    
        [context('user_gestures_training'), context('user_gestures_validation'), context('user_gestures_testing')] = ...
            splitUserDataIndividual(user_folder, path_to_data, ignoreGestures, params.porc_training, params.porc_validation, "packet");
        total_episodes = hyperparams.general_epochs * length(context('user_gestures_training')); % if is general: * num_users;
        q_neural_network = buildQNeuralNetwork(hyperparams, total_episodes);

        do_validation = params.porc_validation > 0;

        [history_episodes_by_epoch, summary, ~] = trainAndValidate(path_to_framework, path_root, q_neural_network, ...
                                            hyperparams.general_epochs, do_validation, context, params.verbose_level-1);

        if do_validation
            accuracy_classification_window = accuracy_classification_window + summary{hyperparams.general_epochs, 2}.classification_window_validation.accuracy;
            accuracy_classification = accuracy_classification + summary{hyperparams.general_epochs, 2}.classification_validation.accuracy;
            accuracy_recognition = accuracy_recognition + summary{hyperparams.general_epochs, 2}.recognition_validation.accuracy;
        else
            accuracy_classification_window = accuracy_classification_window + summary{hyperparams.general_epochs, 1}.classification_window_train.accuracy;
            accuracy_classification = accuracy_classification + summary{hyperparams.general_epochs, 1}.classification_train.accuracy;
            accuracy_recognition = accuracy_recognition + summary{hyperparams.general_epochs, 1}.recognition_train.accuracy;
        end

        ExperimentHelper.saveModel(params.qnn_model_dir_name, q_neural_network, history_episodes_by_epoch, summary, context);

        % Test
        if run_test_as_validation
            if use_same_training_set_as_validation
                context('user_gestures_validation') = context('user_gestures_training');
            end
        end
        [summary, responses] = ExperimentHelper.testModelIndividual(q_neural_network, run_test_as_validation, context, verbose_level-1);
        disp("");
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

%% Visualizations


all_rewards = [];
all_updates = [];

for epoch=1:hyperparams.general_epochs
    all_rewards = [all_rewards, history_episodes_by_epoch{epoch, 1}.history_rewards];
    
    update_costs_by_episode = history_episodes_by_epoch{epoch, 1}.history_update_costs;

    for index_gesture=1:length(update_costs_by_episode)
        costs = update_costs_by_episode{index_gesture};
        all_updates = [all_updates; costs(:)];
    end
    
end
figure(1);
subplot(1,2,1)
plot(all_rewards);
title("Train: Reward");

subplot(1,2,2)
plot(all_updates(500:end));
title("Cost");
