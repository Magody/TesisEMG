function [accuracy_classification_window, accuracy_classification, accuracy_recognition] = ...
    qnnModelFeature(gens, path_to_data, path_root, path_to_framework, verbose_level)

    
    experiment_id = 7;
    experiment_mode = "individual";

    experiments_csv = readtable(path_root + 'ModelingAndExperiments/Experiments/experiments_parameters_QNN.csv');
    params_experiment_row = experiments_csv(experiment_id, :);

    [params, hyperparams] = build_params(params_experiment_row, experiment_mode, verbose_level);
    hyperparams.executeEpisodeEMG = @executeEpisodeEMG;
    hyperparams.customRunEpisodesEMG = @customRunEpisodesEMG;


    gestures_list = ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"];
    ignoreGestures = [];
    classes_num_to_name = getClassNumToName(gestures_list, ignoreGestures);    
    context = generateContext(params, classes_num_to_name);
    
    
    hyperparams.interval_for_learning = gens.interval_for_learning;  % in each episode will learn this n times more or less
    hyperparams.learning_rate = gens.learning_rate;
    hyperparams.gamma = gens.gamma;
    

    hyperparams.sequential_network = Sequential({
        Dense(gens.hidden1_neurons, "kaiming", 40), ...
        Activation("relu"), ...
        Dropout(gens.dropout_rate1), ...
        Dense(gens.hidden2_neurons, "kaiming"), ...
        Activation("relu"), ...
        Dense(6, "kaiming"), ...
    });


    num_users = length(params.list_users);
    accuracy_classification_window = 0;
    accuracy_classification = 0;
    accuracy_recognition = 0;
    % This script is for individual model only
    for user_id=101:101 % num_users
        try
            user_folder = "user"+user_id;  
            [context('user_gestures_training'), context('user_gestures_validation'), context('user_gestures_testing')] = ...
                splitUserDataIndividual(user_folder, path_to_data, ignoreGestures, params.porc_training, params.porc_validation, "packet");
            total_episodes = hyperparams.general_epochs * length(context('user_gestures_training')); % if is general: * num_users;
            q_neural_network = buildQNeuralNetwork(hyperparams, total_episodes);

            do_validation = params.porc_validation > 0;

            [history_episodes_by_epoch, summary, ~] = trainAndValidate(path_to_framework, path_root, q_neural_network, ...
                                                hyperparams.general_epochs, do_validation, context, params.verbose_level-1);

            if do_validation
                accuracy_classification_window = accuracy_classification_window + summary{hyperparams.general_epochs, 2}.classification_window_validation.accuracy;
                accuracy_classification = accuracy_classification + summary{hyperparams.general_epochs, 2}.classification_validation.accuracy;
                accuracy_recognition = accuracy_recognition + summary{hyperparams.general_epochs, 2}.recognition_validation.accuracy;
            else
                accuracy_classification_window = accuracy_classification_window + summary{hyperparams.general_epochs, 1}.classification_window_train.accuracy;
                accuracy_classification = accuracy_classification + summary{hyperparams.general_epochs, 1}.classification_train.accuracy;
                accuracy_recognition = accuracy_recognition + summary{hyperparams.general_epochs, 1}.recognition_train.accuracy;
            end

        catch exception
            fprintf("Error with %s, \n%s\n", user_folder, exception.message + "->" + ...
                    exception.stack(1).name + " line: " + exception.stack(1).line);
        end
    end

    


   

end