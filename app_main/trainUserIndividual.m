function [q_neural_network, training_stats, validation_stats] = trainUserIndividual(params, hyperparams, path_to_framework, path_to_data)
    %{
        Params:
        params.verbose_level = 2;
        params.RepTraining = 150;
        params.RepTesting = 150;
        params.list_users = [208]; % [8 200]; 1:306;
        params.list_users_test = [208]; % [1 2]; 1:306;
        params.rangeDown = 1;
        params.rangeDownTesting = 1;
        params.window_size = 300;
        params.stride = 30;
        params.qnn_model_dir_name = "test.mat";

        Hyperparams:
        addpath(genpath(path_to_framework));
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
    clc;
    close all;
    
    seed_rng = hyperparams.seed_rng;

    %% Libs
    addpath(genpath(path_to_framework));
    addpath(genpath('utils'));
    addpath(genpath('RLSetup'));

    %% Init general parameters
    context = containers.Map();    
    num_users = length(params.list_users);
    num_users_test = length(params.list_users_test);
    context('num_users') = num_users;
    context('num_users_test') = num_users_test;
    context('list_users') = params.list_users;
    context('list_users_test') = params.list_users_test;
    % prepare_environment();    
    assignin('base','RepTraining',  params.RepTraining); % initial value
    context('RepTraining') = params.RepTraining;
    context('RepTesting') = params.RepTesting;
    context('rangeDownTrain') = params.rangeDown;
    context('rangeDownTest') = params.rangeDown;
    context('data_dir') = path_to_data;


    %% Init Hyper parameters and models
    fprintf("Setting hyper parameters and models\n");
    generate_rng(seed_rng);
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

    total_episodes = params.RepTraining * num_users;
    total_episodes_test = params.RepTesting * num_users_test;

    qLearningConfig = QLearningConfig(hyperparams.gamma, hyperparams.epsilon, hyperparams.gameReplayStrategy, hyperparams.experience_replay_reserved_space, total_episodes);
    qLearningConfig.total_episodes_test = total_episodes_test;
    q_neural_network = QNeuralNetwork(sequential_conv_network, hyperparams.sequential_network, ...
                        nnConfig, qLearningConfig, hyperparams.executeEpisodeEMG);    % @executeEpisodeEMGImage 

    q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG);


    %% Train


    fprintf("*****Training with %d users, each one with %d gestures*****\n", num_users, params.RepTraining);

    t_begin = tic;
    for epoch=1:hyperparams.general_epochs
        for index_id_user=1:num_users
            % extracting user vars

            user_real_id = params.list_users(index_id_user);

            user_folder = "user"+user_real_id;

            % just use the feature table in the path to data
            userData = loadUserByNameAndDir(user_folder, path_to_data, false);
            context('user_gestures') = packerByGestures(userData.training, ""); % (randperm(numel(userData.training)));
            
            
            

            context('offset_user') = (index_id_user-1) * params.RepTraining;

            history_episodes_train = q_neural_network.runEpisodes(@getRewardEMG, false, context, params.verbose_level-1);
        end
    end
    t_end = toc(t_begin);
    fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
    
    

    train_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_window_correct'), history_episodes_train('history_classification_window_incorrect'));
    fprintf("Train: Mean accuracy for classification window: %.4f\n", train_metrics_classification_window('accuracy'));

    train_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_class_correct'), history_episodes_train('history_classification_class_incorrect'));
    fprintf("Train: Mean accuracy for classification class: %.4f\n", train_metrics_classification_class('accuracy'));

    train_metrics_classification_recognition = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_recognition_correct'), history_episodes_train('history_classification_recognition_incorrect'));
    fprintf("Train: Mean accuracy for recognition: %.4f\n", train_metrics_classification_recognition('accuracy'));


    %% test

    fprintf("*****Test with %d users, each one with %d gestures*****\n", num_users_test, params.RepTesting);

    userDataTest = loadUserByNameAndDir("user" + params.list_users_test(1), path_to_data, false);
    context('user_gestures') = userDataTest.testing;

    history_episodes_test = q_neural_network.runEpisodes(@getRewardEMG, true, context, params.verbose_level-1);

    fprintf("Test: Mean collected reward %.4f\n", mean(history_episodes_test("history_rewards")));
    test_classif_window_correct = history_episodes_test('history_classification_window_correct');
    test_classif_window_incorrect = history_episodes_test('history_classification_window_incorrect');
    test_metrics_classification_window = getMetricsFromCorrectIncorrect(test_classif_window_correct, test_classif_window_incorrect);
    fprintf("Test: Mean accuracy for classification window: %.4f\n", test_metrics_classification_window('accuracy'));


    validation_classif_correct = history_episodes_test('history_classification_class_correct');
    validation_classif_incorrect = history_episodes_test('history_classification_class_incorrect');
    validation_metrics_classification_class_accuracy = getMetricsFromCorrectIncorrect(validation_classif_correct, validation_classif_incorrect);
    fprintf("Validation: Mean accuracy for classification class: %.4f\n", validation_metrics_classification_class_accuracy('accuracy'));

    validation_recog_correct = history_episodes_test('history_classification_recognition_correct');
    validation_recog_incorrect = history_episodes_test('history_classification_recognition_incorrect');
    validation_metrics_classification_recognition_accuracy = getMetricsFromCorrectIncorrect(validation_recog_correct, validation_recog_incorrect);
    fprintf("Test: Mean accuracy for classification recog: %.4f\n", validation_metrics_classification_recognition_accuracy('accuracy'));


    %% save model
    
    training_stats = struct("classification_by_window", train_metrics_classification_window("classification_by_window"), ...
                            "classification", train_metrics_classification_class("classification"), ...
                            "recognition", train_metrics_classification_recognition("recognition"));

    validation_stats = struct("classification_by_window", test_metrics_classification_window("classification_by_window"), ...
                            "classification", validation_metrics_classification_class_accuracy("classification"), ...
                            "recognition", validation_metrics_classification_recognition_accuracy("recognition"));

    save(params.qnn_model_dir_name, "q_neural_network", "training_stats", "validation_stats");

    

end