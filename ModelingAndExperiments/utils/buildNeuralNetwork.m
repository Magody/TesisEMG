function neural_network = buildNeuralNetwork(hyperparams)
    
    % addpath(genpath(path_to_framework));
    
    seed_rng = hyperparams.seed_rng;
    generate_rng(seed_rng);    

    nnConfig = NNConfig(hyperparams.inner_epochs, hyperparams.learning_rate, hyperparams.batch_size, hyperparams.loss_type);
    nnConfig.decay_rate_alpha = hyperparams.decay_rate_alpha;

    neural_network = NeuralNetwork(hyperparams.sequential_network, nnConfig);
   
    
end

