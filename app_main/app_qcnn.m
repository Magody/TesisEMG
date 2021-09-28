%% Clean up
clc;
clear all; %#ok<CLALL>
close all;

seed_rng = 44;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
addpath(genpath(path_to_framework));
addpath(genpath('utils'));
addpath('RLSetup');

%% Init general parameters
context = containers.Map();
verbose_level = 10;
RepTraining = 20;
RepTesting = 15;


list_users = [8]; % [8 200]; 1:306;
list_users_test = [1]; % [1 2]; 1:306;
num_users = length(list_users);
num_users_test = length(list_users_test);

rangeDown = 1;
rangeDownTesting = 1;

prepare_environment('/home/magody/programming/MATLAB/tesis/Data/', verbose_level-1);
assignin('base','RepTraining',  RepTraining); % initial value
context('RepTraining') = RepTraining;
context('RepTesting') = RepTesting;

context('num_users') = num_users;
context('num_users_test') = num_users_test;
context('rangeDownTrain') = rangeDown;
context('rangeDownTest') = rangeDown;
context('list_users') = list_users;
context('list_users_test') = list_users_test;
context('data_dir') = '/home/magody/programming/MATLAB/tesis/Data/';



%% Generating orientation (slow section)
fprintf("Generating orientation...\n");
assignin('base','packetEMG',     false); 
Code_0(rangeDown, '/home/magody/programming/MATLAB/tesis/Data/');
orientation      = evalin('base', 'orientation');
dataPacket = evalin('base','dataPacket');
fprintf("Orientation generated\n");
    
%% Init Hyper parameters and models
fprintf("Setting hyper parameters and models\n");
generate_rng(seed_rng);
context('interval_for_learning') = 3;  % in each episode will learn this n times more or less
window_size = 300;
stride = 30;

%{
use_channels = false;
height = 10;
context("image_config") = struct("width", 1000, "height", height, "merge", ~use_channels);

shape_input = [height, window_size, 1];  % shape_input = [1, window_size, 8];
if use_channels
    shape_input = [height, window_size, 8];
end
%}
shape_input = [1, window_size, 8];

total_episodes = RepTraining * num_users;
total_episodes_test = RepTesting * num_users_test;
epochs = 1; % epochs inside each NN
learning_rate = 0.001;
batch_size = 64;
gamma = 0.1;
epsilon = 1;
decay_rate_alpha = 0.1;
gameReplayStrategy = 1;
experience_replay_reserved_space = 20;
loss_type = "mse";
rewards = struct('correct', 1, 'incorrect', -1);

context('window_size') = window_size;
context('stride') = stride;
context('rewards') = rewards;
assignin('base','WindowsSize',  window_size);
assignin('base','Stride',  stride);

context('orientation') = orientation;
context('dataPacket') = dataPacket;


sequential_conv_network = Sequential({
    Convolutional([1, 3], 8, 0, 1, shape_input), ...
    Activation("relu"), ....
    Pooling("max", [1, 2]), ...
    Reshape(), ...
});

input_dense = prod(sequential_conv_network.shape_output);% if convolutional network exist, sequential_conv_network.shape_output;

sequential_network = Sequential({
    Dense(64, "kaiming", input_dense), ...
    Activation("relu"), ...
    Dense(64, "kaiming"), ...
    Activation("relu"), ...
    Dense(6, "xavier"), ...
});

nnConfig = NNConfig(epochs, learning_rate, batch_size, loss_type);
nnConfig.decay_rate_alpha = decay_rate_alpha;

qLearningConfig = QLearningConfig(gamma, epsilon, gameReplayStrategy, experience_replay_reserved_space, total_episodes);
qLearningConfig.total_episodes_test = total_episodes_test;
q_neural_network = QNeuralNetwork(sequential_conv_network, sequential_network, ...
                    nnConfig, qLearningConfig, @executeEpisodeEMG);    % @executeEpisodeEMGImage 

q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG);

%% Train

fprintf("*****Training with %d users, each one with %d gestures*****\n", num_users, RepTraining);
        
t_begin = tic;
history_episodes_train = q_neural_network.runEpisodes(@getRewardEMG, false, context, verbose_level-1);

t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

%% plot results
% disp(history_episodes_train('history_gestures_name'));
figure(1);
subplot(1,2,1)
plot(history_episodes_train('history_rewards'));
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




%% TSNE, dimension reduction. Table of features and plots

class_names = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"]; 
valid_replay = getCellsNotEmpty(q_neural_network.gameReplay);

len_gamereplay = length(q_neural_network.gameReplay);

dataX = [];
dataY = [];

input_dim = length(q_neural_network.shape_input);

index_valid = 0;

for numExample=1:len_gamereplay
    
    reward_er = valid_replay{numExample}.reward;
    
    if reward_er < 0
        % continue;
    end
    
    index_valid = index_valid + 1;
    
    s = valid_replay{numExample}.state;    
    action_er = valid_replay{numExample}.action;

    if input_dim == 1
        dataX(:, index_valid) = s;
    elseif input_dim == 2
        dataX(:, :, index_valid) = s;
    elseif input_dim == 3
        dataX(:, :, :, index_valid) = s;
    end

    dataY(index_valid, :) = sparse_one_hot_encoding(action_er, length(class_names));

end


fprintf("Generating table of features\n");
len_data_train = size(dataX, 4);
features_matrix = zeros([len_data_train, q_neural_network.sequential_conv_network.shape_output]);
classes = {};




for index_data=1:len_data_train
    
    x = dataX(:, :, :, index_data);
    [~, y] = max(dataY(index_data, :));  % argmax
    features = q_neural_network.sequential_conv_network.forward(x);
    
    
    classes{index_data, 1} = char(class_names(y));
    features_matrix(index_data, :) = features';
            
end

%% TSNE plots
% features_matrix = table2array(features_table);

options = containers.Map();
options('limit_samples') = 10000;
options('plot_point_size') = 20;
options('include3D') = false;
options('save') = false;

options('dir') = 'figures/';
options('algorithms') =  [ ...
    struct('distance', 'euclidean','plot', struct('title', 'Euclidean')), ...
    % struct('distance', 'chebychev','plot', struct('title', 'Chebychev')), ...
];

history_tsne = generateTSNE(features_matrix, classes, options, verbose_level-1);
k = keys(history_tsne);
v = values(history_tsne);
for i=1:length(k)
    fprintf("Results for algorithm %s: ", k{i});
    disp(v{i});
end
