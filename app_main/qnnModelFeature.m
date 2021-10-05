function [accuracy_train, accuracy_test] = qnnModelFeature(params, path_to_data, verbose_level)

    context = containers.Map();
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
    context('interval_for_learning') = params.interval_for_learning;  % in each episode will learn this n times more or less
    window_size = 300;
    stride = 30;
    context('tabulation_mode') = 2;
    context('is_preprocessed') = true;
    context('noGestureDetection') = false;

    epochs = params.epochs_nn; % epochs inside each NN
    learning_rate = params.learning_rate;
    batch_size = params.batch_size;
    gamma = params.gamma;
    epsilon = 1;
    decay_rate_alpha = params.decay_rate_alpha;
    gameReplayStrategy = 1;
    experience_replay_reserved_space = params.experience_replay_reserved_space;
    loss_type = "mse";
    rewards = struct('correct', 1, 'incorrect', -1);

    context('window_size') = window_size;
    context('stride') = stride;
    context('rewards') = rewards;
    assignin('base','WindowsSize',  window_size);
    assignin('base','Stride',  stride);


    sequential_conv_network = Sequential({});

    sequential_network = Sequential({
        Dense(params.hidden1_neurons, "kaiming", 40), ...
        Activation("relu"), ...
        Dropout(params.dropout_rate1), ...
        Dense(params.hidden2_neurons, "kaiming"), ...
        Activation("relu"), ...
        Dropout(params.dropout_rate2), ...
        Dense(6, "xavier"), ...
    });

    nnConfig = NNConfig(epochs, learning_rate, batch_size, loss_type);
    nnConfig.decay_rate_alpha = decay_rate_alpha;

    list_users = 208; % [8 200]; 1:306;
    list_users_test = 306; % [1 2]; 1:306;
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
    if verbose_level > 0
        fprintf("*****Training with %d users, each one with %d gestures*****\n", num_users, RepTraining);
    end
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
    
    train_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_window_correct'), history_episodes_train('history_classification_window_incorrect'));
    train_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_class_correct'), history_episodes_train('history_classification_class_incorrect'));
    train_metrics_classification_recognition = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_recognition_correct'), history_episodes_train('history_classification_recognition_incorrect'));

    accuracy_train = (train_metrics_classification_window('accuracy')*100 + ...
        train_metrics_classification_class('accuracy')*100 + ...
        train_metrics_classification_recognition('accuracy')*100)/3;
    
    
    history_episodes_test = q_neural_network.runEpisodes(@getRewardEMG, true, context, verbose_level-1);

    
    test_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_window_correct'), history_episodes_test('history_classification_window_incorrect'));
    test_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_class_correct'), history_episodes_test('history_classification_class_incorrect'));
    test_metrics_classification_recognition = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_recognition_correct'), history_episodes_test('history_classification_recognition_incorrect'));

            
    accuracy_test = (test_metrics_classification_window('accuracy')*100 + ...
        test_metrics_classification_class('accuracy')*100 + ...
        test_metrics_classification_recognition('accuracy')*100)/3;
    


   

end