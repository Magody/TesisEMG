function helper_individual_test(path_to_framework, path_root, group_folder, ...
    experiment_id, model_folder, ignoreGestures)
path_root = string(path_root);
model_folder = string(model_folder);

addpath(genpath(path_to_framework));
addpath(path_root + "ModelingAndExperiments/utils")
addpath(path_root + "ModelingAndExperiments/learning")
addpath(path_root + "ModelingAndExperiments/RLSetup")
addpath(path_root + "ModelingAndExperiments/Experiments")
addpath(genpath(path_root + "GeneralLib"));

path_output = path_root + "ModelingAndExperiments/models/" + model_folder + "/";  % test_low_umbral


% processing processingTest
data_path = ['Data/', group_folder, '/'];
path_to_data_for_train = horzcat(char(path_root),data_path); % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';
path_to_data_for_testing = horzcat(char(path_root),data_path); % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';

jsonName=horzcat(char(path_output), 'responses.json');
version = 'testing';

%% set parameters
verbose_level = 2;
experiment_mode = "individual";


experiments_csv = readtable(path_root + 'ModelingAndExperiments/Experiments/experiments_parameters_QNN.csv');
params_experiment_row = experiments_csv(experiment_id, :);
        
[params, hyperparams] = build_params(params_experiment_row, experiment_mode, verbose_level);
hyperparams.executeEpisodeEMG = @executeEpisodeEMG;
hyperparams.customRunEpisodesEMG = @customRunEpisodesEMG;

gestures_list = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"];
% ignoreGestures = [];
classes_num_to_name = getClassNumToName(gestures_list, ignoreGestures);    
context = generateContext(params, classes_num_to_name);

users = dir(path_to_data_for_testing);
users = users(3:end);
num_users = length(users);

%% Test

run_test_as_validation = false;
use_same_training_set_as_validation = true;

accuracy_classification = 0;
accuracy_recognition = 0;

Rep = 150;  % below it is not controlled, is with length(responses)

results = struct();

counter = 0;

for i=1:num_users  % 3 for user100
    try
        user_folder = users(i).name;
		
		disp(i);
        
		
		%{
		if user_folder == "user115" || user_folder == "user209"
            continue;
        end		
		just to check if is ok
		if user_folder ~= "user1"
			continue;
		end
		
		%}
		
		counter = counter + 1;
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
       accuracy_classification/counter, accuracy_recognition/counter); 
end

%% Export  in json
jsonFormat(jsonName, results);