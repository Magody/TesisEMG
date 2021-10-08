
clc;
clear all;
close all;

%% Libs
verbose_level = 2;

path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
path_to_data_for_train = '/home/magody/programming/MATLAB/tesis/Data/preprocessingTest/'; % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';
path_to_data_for_test = '/home/magody/programming/MATLAB/tesis/Data/preprocessingTest/'; % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';

addpath('LabEPN');
addpath('learning');
addpath(genpath(path_to_framework));
addpath(genpath('utils'));
addpath(genpath('RLSetup'));

%% set parameters
params.verbose_level = verbose_level-1;
params.RepTraining = 150;
params.RepValidation = 0;
params.RepTesting = 150;
params.debug_steps = 1;


params.rangeDown = 1;
params.rangeDownValidation = 1;
params.rangeDownTest = 1;
params.window_size = 300;
params.stride = 30;
params.debug_steps = 1;
hyperparams.seed_rng = 44;
hyperparams.interval_for_learning = 10;
hyperparams.inner_epochs = 5; % epochs inside each NN
hyperparams.learning_rate = 0.001;
hyperparams.batch_size = 128;
hyperparams.gamma = 0.1;
hyperparams.epsilon = 1;
hyperparams.decay_rate_alpha = 0.1;
hyperparams.gameReplayStrategy = 1;
hyperparams.experience_replay_reserved_space = 100;
hyperparams.loss_type = "mse";
hyperparams.rewards = struct('correct', 1, 'incorrect', -1);
    
hyperparams.executeEpisodeEMG = @executeEpisodeEMG;
hyperparams.general_epochs = 5;

packet_data = orderfields(dir(path_to_data_for_train));
len_users = length(packet_data) - 2; % length(packet_data) - 2; 10;

%% Train and validate
total1 = 0;
total2 = 0;
total3 = 0;
t_begin = tic;
for k=1:len_users
    
    % ensure is new network every time
    hyperparams.sequential_network = Sequential({
        Dense(40, "kaiming", 40), ...
        Activation("relu"), ...
        Dense(40, "kaiming"), ...
        Activation("relu"), ...
        Dense(6, "xavier"), ...
    });
    
    user_folder = packet_data(2+k).name;
    user_id = str2double(user_folder(5:end));
    
    params.list_users = [user_id];
    params.list_users_validation = [user_id];
    params.qnn_model_dir_name = "models/" + user_folder + ".mat";
    
    try
        [~, ~, summary, do_validation] = trainUserIndividual(params, hyperparams, path_to_framework, path_to_data_for_train);

        if do_validation
            total1 = total1 + summary{hyperparams.general_epochs, 2}.classification_window_validation.accuracy;
            total2 = total2 + summary{hyperparams.general_epochs, 2}.classification_validation.accuracy;
            total3 = total3 + summary{hyperparams.general_epochs, 2}.recognition_validation.accuracy;
        else
            total1 = total1 + summary{hyperparams.general_epochs, 1}.classification_window_train.accuracy;
            total2 = total2 + summary{hyperparams.general_epochs, 1}.classification_train.accuracy;
            total3 = total3 + summary{hyperparams.general_epochs, 1}.recognition_train.accuracy;
        
        end
        
    catch ME
        fprintf("Train: Error with %s\n", user_folder);
    end

end
t_end = toc(t_begin);
if params.verbose_level > 0
    fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
    fprintf("Final class: %.4f, Final recognition: %.4f\n", total2/len_users, total3/len_users); 
end

%% Test

Rep = 150;
context = containers.Map();
context('tabulation_mode') = 2;
context('is_preprocessed') = true;
context('noGestureDetection') = false;
context('window_size') = params.window_size;
context('stride') = params.stride;
context('rewards') = hyperparams.rewards;
assignin('base','WindowsSize',  params.window_size);
assignin('base','Stride',  params.stride);


jsonName='responses.json';
type_execution = 3;





version = 'testing';

results = struct();

total1 = 0;
total2 = 0;
total3 = 0;

run_as_validation = false;
if run_as_validation
    type_execution = 2;
    context('RepValidation') = params.RepTesting;
    context('rangeDownValidation') = params.rangeDownTest;
else
    context('RepTesting') = params.RepTesting;
    context('rangeDownTest') = params.rangeDownTest;
end

for k=1:len_users
    
    user_folder = packet_data(2+k).name;
    user_id = str2double(user_folder(5:end));
    
    qnn_model = load("models/" + user_folder + ".mat");
    
    userData = loadUserByNameAndDir(user_folder, path_to_data_for_test, false);
    dataset_part1 = userData.training;
    dataset_part2 = userData.testing;  
    
    if run_as_validation
        context('user_gestures_validation') = [dataset_part2];  % [dataset_part1, dataset_part2]
    else
        context('user_gestures_test') = [dataset_part2];  % [dataset_part1, dataset_part2]
    end
    clear dataset_part1;
    clear dataset_part2;   
    clear userData;   
    
    params.list_users_validation = [user_id];
    
    context('offset_user') = 0;
    
    try
        
        history_test = qnn_model.q_neural_network.runEpisodes(@getRewardEMG, type_execution, context, params.verbose_level-1);

        history_responses = history_test.history_responses;

        if run_as_validation
            [classification_window_test, classification_test, recognition_test] = Experiment.getEpisodesEMGMetrics(history_test);
            total1 = total1 + classification_window_test.accuracy;
            total2 = total2 + classification_test.accuracy;
            total3 = total3 + recognition_test.accuracy;
            if classification_test.accuracy < 0.92
                fprintf("%s->Test accuracy: [%.4f, %.4f, %.4f]\n", user_folder, ...
                    classification_window_test.accuracy, classification_test.accuracy, recognition_test.accuracy);
            end
        end
        
        % fprintf("%s %d\n", user_folder, length(history_responses{150}.vectorOfTimePoints));


        for j =  0:Rep-1
            sample = sprintf('idx_%d',j);

            results.(version).(user_folder).class.(sample)                  = history_responses{j+1}.class ;
            results.(version).(user_folder).vectorOfLabels.(sample)         = history_responses{j+1}.vectorOfLabels;
            results.(version).(user_folder).vectorOfTimePoints.(sample)     = history_responses{j+1}.vectorOfTimePoints;
            results.(version).(user_folder).vectorOfProcessingTime.(sample) = history_responses{j+1}.vectorOfProcessingTimes;

        end
    catch ME
        fprintf("Test: Error with %s", user_folder);
    end
    
    
end

if run_as_validation
   fprintf("Final class: %.4f, Final recognition: %.4f\n", total2/len_users, total3/len_users); 
end

%% Export  in json
jsonFormat(jsonName, version, Rep, results);



