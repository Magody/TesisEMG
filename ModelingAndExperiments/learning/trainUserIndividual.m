function [q_neural_network, history_episodes_by_epoch, summary, do_validation] = ...
            trainUserIndividual(params, hyperparams, ...
            path_to_framework, path_to_data, context_initial)
    % Train with a user, and validate with the same or another user
    %{
        path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
        path_to_data = '/home/magody/programming/MATLAB/tesis/Data/preprocessing/'; % 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\preprocessing\'; % '/home/magody/programming/MATLAB/tesis/Data/preprocessing/';


        addpath(genpath(path_to_framework));
        params.verbose_level = 2;
        params.RepTraining = 240;
        params.RepValidation = 60;
        params.list_users = [208]; % [8 200]; 1:306;
        params.list_users_validation = [208]; % [1 2]; 1:306;
        params.rangeDown = 1;
        params.rangeDownValidation = 1;
        params.window_size = 300;
        params.stride = 30;
        params.qnn_model_dir_name = "models/user208.mat";

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

        hyperparams.sequential_network = Sequential({
                Dense(40, "kaiming", 40), ...
                Activation("relu"), ...
                Dense(40, "kaiming"), ...
                Activation("relu"), ...
                Dense(6, "xavier"), ...
            });


        hyperparams.executeEpisodeEMG = @executeEpisodeEMG;


        hyperparams.general_epochs = 5;
    %}

    %% Clean up
    close all;
    
    seed_rng = hyperparams.seed_rng;
    generate_rng(seed_rng);

    %% Libs
    addpath(genpath(path_to_framework));
    addpath(genpath('utils'));
    addpath(genpath('RLSetup'));

    %% Init general parameters
    context = context_initial;
    
    
    num_users = length(params.list_users);
    num_users_validation = length(params.list_users_validation);
    context('num_users') = num_users;
    context('num_users_validation') = num_users_validation;
    context('list_users') = params.list_users;
    context('list_users_validation') = params.list_users_validation;
    
    % prepare_environment();    
    assignin('base','RepTraining',  params.RepTraining); % initial value
    context('RepTraining') = params.RepTraining;
    context('RepValidation') = params.RepValidation;
    context('rangeDownTrain') = params.rangeDown;
    context('rangeDownValidation') = params.rangeDownValidation;
    context('data_dir') = path_to_data;
    
    %% data set
    user_real_id = params.list_users(1);
    user_folder = "user"+user_real_id;
    userData = loadUserByNameAndDir(user_folder, path_to_data, false);
    dataset_part1 = packerByGestures(userData.training, params.ignoreGestures);
    
    dataset_part2 = {};
    if params.RepValidation ~= 0
        dataset_part2 = packerByGestures(userData.testing, params.ignoreGestures);    
    end
    
    do_validation = ~isempty(dataset_part2) || (params.RepTraining < 150 && (params.RepTraining + params.RepValidation) <= 150);
    
    dataset_complete = [dataset_part1, dataset_part2];
    
    context('user_gestures_train') = dataset_complete(1:params.RepTraining-25);
    
    if do_validation
        context('user_gestures_validation') = dataset_complete((params.RepTraining+1-25):(params.RepTraining+params.RepValidation-25));
    end
    
    clear dataset_part1;
    clear dataset_part2;   
    clear dataset_complete;  
    clear userData;   

    %% Init Hyper parameters and models
    context('interval_for_learning') = hyperparams.interval_for_learning;  % in each episode will learn this n times more or less
    
    context('tabulation_mode') = 2;
    context('is_preprocessed') = true;
    context('noGestureDetection') = false;
    context('window_size') = params.window_size;
    context('stride') = params.stride;
    context('rewards') = hyperparams.rewards;
    assignin('base','WindowsSize',  params.window_size);
    assignin('base','Stride',  params.stride);

    sequential_conv_network = Sequential({});
    

    nnConfig = NNConfig(hyperparams.inner_epochs, hyperparams.learning_rate, hyperparams.batch_size, hyperparams.loss_type);
    nnConfig.decay_rate_alpha = hyperparams.decay_rate_alpha;

    total_episodes = hyperparams.general_epochs * length(context('user_gestures_train')) * num_users;

    qLearningConfig = QLearningConfig(hyperparams.gamma, hyperparams.epsilon, hyperparams.gameReplayStrategy, hyperparams.experience_replay_reserved_space, total_episodes);

    q_neural_network = QNeuralNetwork(sequential_conv_network, hyperparams.sequential_network, ...
                        nnConfig, qLearningConfig, hyperparams.executeEpisodeEMG);    % @executeEpisodeEMGImage 

    q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG);


    %% Train

    if params.verbose_level > 0
        fprintf("*****Training with %s, with %d gestures*****\n", user_folder, params.RepTraining);
    end
    history_episodes_by_epoch = cell([hyperparams.general_epochs, 2]);
    summary = cell([hyperparams.general_epochs, 2]);
    t_begin = tic;
    print_step = ceil(hyperparams.general_epochs/params.debug_steps);
    for epoch=1:hyperparams.general_epochs

        context('offset_user') = 0;
        context('global_epoch') = epoch;

        % Train
        history_episodes_by_epoch{epoch, 1} = q_neural_network.runEpisodes(@getRewardEMG, 1, context, params.verbose_level-1);
        

        [classification_window_train, classification_train, recognition_train] = Experiment.getEpisodesEMGMetrics(history_episodes_by_epoch{epoch, 1});
        
        summary{epoch, 1} = struct("classification_window_train", classification_window_train, ...
                                   "classification_train", classification_train, ...
                                   "recognition_train", recognition_train);
                               
        if do_validation
            % validation
            history_episodes_by_epoch{epoch, 2} = q_neural_network.runEpisodes(@getRewardEMG, 2, context, params.verbose_level-1);

            [classification_window_validation, classification_validation, recognition_validation] = Experiment.getEpisodesEMGMetrics(history_episodes_by_epoch{epoch, 2});

            summary{epoch, 2} = struct("classification_window_validation", classification_window_validation, ...
                                       "classification_validation", classification_validation, ...
                                       "recognition_validation", recognition_validation);
        end

        if params.verbose_level > 0
            if mod(epoch, print_step) == 0 || epoch == hyperparams.general_epochs
                if do_validation
                    fprintf("->Epoch %d | Train accuracy: [%.4f, %.4f, %.4f], Validation accuracy: [%.4f, %.4f, %.4f]\n", epoch, ...
                        classification_window_train.accuracy, classification_train.accuracy, recognition_train.accuracy, ...
                        classification_window_validation.accuracy, classification_validation.accuracy, recognition_validation.accuracy);
                else
                    fprintf("->Epoch %d | Train accuracy: [%.4f, %.4f, %.4f]\n", epoch, ...
                        classification_window_train.accuracy, classification_train.accuracy, recognition_train.accuracy);
                end
            end
        end
        
    end
    t_end = toc(t_begin);
    if params.verbose_level > 1
        fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
    end

    %% save model
    
    classes_num_to_name = context('classes_num_to_name');
    classes_name_to_num = context('classes_name_to_num');

    
    save(params.qnn_model_dir_name, "q_neural_network", ...
                                    "history_episodes_by_epoch", ...
                                    "summary", ...
                                    "classes_num_to_name", ...
                                    "classes_name_to_num");

    

end