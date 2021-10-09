path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
path_to_data = '/home/magody/programming/MATLAB/tesis/Data/preprocessing/'; % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';


addpath(genpath(path_to_framework));
params.verbose_level = 2;

params.debug_steps = 10;

params.RepTraining = 150;
params.RepValidation = 150;
params.list_users = [101]; % [8 200]; 1:306;
params.list_users_validation = [101]; % [1 2]; 1:306;
params.rangeDown = 1;
params.rangeDownValidation = 1;
params.window_size = 300;
params.stride = 30;
params.qnn_model_dir_name = "debug.mat";



hyperparams.seed_rng = 44;
hyperparams.interval_for_learning = 10;
hyperparams.inner_epochs = 5; % epochs inside each NN
hyperparams.learning_rate = 0.001;
hyperparams.batch_size = 128;
hyperparams.gamma = 0.1;
hyperparams.epsilon = 1;
hyperparams.decay_rate_alpha = 0.6;
hyperparams.gameReplayStrategy = 1;
hyperparams.experience_replay_reserved_space = 100;
hyperparams.loss_type = "mse";
hyperparams.rewards = struct('correct', 1, 'incorrect', -1);

hyperparams.sequential_network = Sequential({
        Dense(16, "kaiming", 40), ...
        Activation("relu"), ...
        Dense(16, "kaiming"), ...
        Activation("relu"), ...
        Dense(6, "kaiming"), ...
    });
    
    
hyperparams.executeEpisodeEMG = @executeEpisodeEMG;
hyperparams.general_epochs = 5;

[q_neural_network, history_episodes_by_epoch, summary, do_validation] = trainUserIndividual(params, hyperparams, path_to_framework, path_to_data);

%% graph


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
