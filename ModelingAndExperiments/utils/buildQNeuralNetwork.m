function q_neural_network = buildQNeuralNetwork(hyperparams, total_episodes)
    
    % addpath(genpath(path_to_framework));
    
    seed_rng = hyperparams.seed_rng;
    generate_rng(seed_rng);    

    nnConfig = NNConfig(hyperparams.inner_epochs, hyperparams.learning_rate, hyperparams.batch_size, hyperparams.loss_type);
    nnConfig.decay_rate_alpha = hyperparams.decay_rate_alpha;

    
    qLearningConfig = QLearningConfig(hyperparams.gamma, hyperparams.epsilon, ...
        hyperparams.gameReplayStrategy, hyperparams.experience_replay_reserved_space, ...
        total_episodes, hyperparams.interval_for_learning, ...
        hyperparams.rewards);

    q_neural_network = QNeuralNetwork(hyperparams.sequential_conv_network, hyperparams.sequential_network, ...
                        nnConfig, qLearningConfig, hyperparams.executeEpisodeEMG);

    q_neural_network.setCustomRunEpisodes(hyperparams.customRunEpisodesEMG);

    
end

