function learn(app, use_dataset)
    global qnn_online context_online
    global known_dataset_training
    global path_to_framework path_root
    global extra_dataset
    if use_dataset
        
        dataset = [extra_dataset, known_dataset_training];
    
        dataset = dataset(randperm(numel(dataset)));
                    
        do_validation = false;
        context_online('user_gestures_training') = dataset;
        
        general_epochs = 10;
        qnn_online.alpha = qnn_online.nnConfig.learning_rate;
     
        trainAndValidate(path_to_framework, path_root, qnn_online, ...
                                            general_epochs, do_validation, context_online, 1);
    else
        verbose_level = 0;
        t = 3;
        
        % qnn_online.alpha = qnn_online.nnConfig.learning_rate/(1 + qnn_online.nnConfig.decay_rate_alpha * t);
        qnn_online.alpha = qnn_online.nnConfig.learning_rate;
        qnn_online.updateQNeuralNetworkTarget();
        
        qnn_online.learnFromExperienceReplay(t, verbose_level-1);
    end
end