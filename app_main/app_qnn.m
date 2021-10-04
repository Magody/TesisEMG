%% Clean up
clc;
clear all; %#ok<CLALL>
close all;

seed_rng = 44;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
path_to_data = '/home/magody/programming/MATLAB/tesis/Data/preprocessing/'; % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';
addpath(genpath(path_to_framework));
addpath(genpath('utils'));
addpath(genpath('RLSetup'));

%% Init general parameters
context = containers.Map();
verbose_level = 2;
RepTraining = 150;
RepTesting = 150;



rangeDown = 1;
rangeDownTesting = 1;

prepare_environment(path_to_data, verbose_level-1);
assignin('base','RepTraining',  RepTraining); % initial value
context('RepTraining') = RepTraining;
context('RepTesting') = RepTesting;

context('rangeDownTrain') = rangeDown;
context('rangeDownTest') = rangeDown;
context('data_dir') = path_to_data;

    
%% Init Hyper parameters and models
fprintf("Setting hyper parameters and models\n");
generate_rng(seed_rng);
context('interval_for_learning') = 10;  % in each episode will learn this n times more or less
window_size = 300;
stride = 30;

context('tabulation_mode') = 2;
context('is_preprocessed') = true;
context('noGestureDetection') = false;

epochs = 3; % epochs inside each NN
learning_rate = 0.0003;
batch_size = 128;
gamma = 0.1;
epsilon = 1;
decay_rate_alpha = 0.1;
gameReplayStrategy = 1;
experience_replay_reserved_space = 100;
loss_type = "mse";
rewards = struct('correct', 1, 'incorrect', -1);

context('window_size') = window_size;
context('stride') = stride;
context('rewards') = rewards;
assignin('base','WindowsSize',  window_size);
assignin('base','Stride',  stride);


sequential_conv_network = Sequential({});

sequential_network = Sequential({
    Dense(40, "kaiming", 40), ...
    Activation("relu"), ...
    Dense(40, "kaiming"), ...
    Activation("relu"), ...
    Dense(6, "xavier"), ...
});

nnConfig = NNConfig(epochs, learning_rate, batch_size, loss_type);
nnConfig.decay_rate_alpha = decay_rate_alpha;

list_users = [1]; % [8 200]; 1:306;
list_users_test = [1]; % [1 2]; 1:306;
num_users = length(list_users);
num_users_test = length(list_users_test);
context('num_users') = num_users;
context('num_users_test') = num_users_test;
context('list_users') = list_users;
context('list_users_test') = list_users_test;

total_episodes = RepTraining * num_users;
total_episodes_test = RepTesting * num_users_test;

qLearningConfig = QLearningConfig(gamma, epsilon, gameReplayStrategy, experience_replay_reserved_space, total_episodes);
qLearningConfig.total_episodes_test = total_episodes_test;
q_neural_network = QNeuralNetwork(sequential_conv_network, sequential_network, ...
                    nnConfig, qLearningConfig, @executeEpisodeEMG);    % @executeEpisodeEMGImage 

q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG);


% % Train



fprintf("*****Training with %d users, each one with %d gestures*****\n", num_users, RepTraining);
        
t_begin = tic;

for index_id_user=1:num_users
    % extracting user vars

    user_real_id = list_users(index_id_user);

    user_folder = "user"+user_real_id;

    % just use the feature table in the path to data
    userData = loadUserByNameAndDir(user_folder, path_to_data, false);
    context('user_gestures') = userData.training(randperm(numel(userData.training)));
    
    context('offset_user') = (index_id_user-1) * RepTraining;

    history_episodes_train = q_neural_network.runEpisodes(@getRewardEMG, false, context, verbose_level-1);
end
t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

% % plot results
% disp(history_episodes_train('history_gestures_name'));
figure(1);
subplot(1,2,1)
history_rewards = history_episodes_train('history_rewards')';
plot(history_rewards(:));
title("Train: Reward");

subplot(1,2,2)
linear_update_costs = [];
update_costs_by_episode = history_episodes_train('history_update_costs');

for i=1:length(update_costs_by_episode)
    costs = update_costs_by_episode{i};
    linear_update_costs = [linear_update_costs; costs(:)]; 
end
plot(linear_update_costs);
title("Cost");

%{
subplot(2,3,4)
train_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_window_correct'), history_episodes_train('history_classification_window_incorrect'));
plot(train_metrics_classification_window('accuracy_by_t'));
title("Acc window");
fprintf("Train: Mean accuracy for classification window: %.4f\n", train_metrics_classification_window('accuracy'));
 
subplot(2,3,5)
train_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_class_correct'), history_episodes_train('history_classification_class_incorrect'));
plot(train_metrics_classification_class('accuracy_by_t'));
title("Acc classification");
fprintf("Train: Mean accuracy for classification class: %.4f\n", train_metrics_classification_class('accuracy'));

%}
train_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_window_correct'), history_episodes_train('history_classification_window_incorrect'));
fprintf("Train: Mean accuracy for classification window: %.4f\n", train_metrics_classification_window('accuracy'));

train_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_class_correct'), history_episodes_train('history_classification_class_incorrect'));
fprintf("Train: Mean accuracy for classification class: %.4f\n", train_metrics_classification_class('accuracy'));

% % test
fprintf("*****Test with %d users, each one with %d gestures*****\n", num_users_test, RepTesting);

history_episodes_test = q_neural_network.runEpisodes(@getRewardEMG, true, context, verbose_level-1);

fprintf("Test: Mean collected reward %.4f\n", mean(history_episodes_test("history_rewards")));
test_classif_window_correct = history_episodes_test('history_classification_window_correct');
test_classif_window_incorrect = history_episodes_test('history_classification_window_incorrect');
test_metrics_classification_window = getMetricsFromCorrectIncorrect(test_classif_window_correct, test_classif_window_incorrect);
fprintf("Test: Mean accuracy for classification window: %.4f\n", test_metrics_classification_window('accuracy'));


test_classif_correct = history_episodes_test('history_classification_class_correct');
test_classif_incorrect = history_episodes_test('history_classification_class_incorrect');
test_metrics_classification_class_accuracy = sum(test_classif_correct)/sum(test_classif_correct+test_classif_incorrect);
fprintf("Test: Mean accuracy for classification class: %.4f\n", test_metrics_classification_class_accuracy);

