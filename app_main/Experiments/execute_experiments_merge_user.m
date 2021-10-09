function execute_experiments_merge_user(model_name, ...
    path_to_framework, path_to_data, verbose_level, ...
    executeEpisodeEMG, noGestureDetection, initial_row_position, is_preprocessed, ...
    experiment_ids)

    model_name = string(model_name);
    path_to_framework = string(path_to_framework);

    %{
    execute_experiments_merge_user("test_merge", ...
    "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework", ...
    '/home/magody/programming/MATLAB/tesis/Data/preprocessing/', 2, ...
    @executeEpisodeEMG, false, 0, true, [14, 14]);
    
    execute_experiments_per_user("test_merge_cnn", ...
    "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework", ...
    '/home/magody/programming/MATLAB/tesis/Data/', 2, ...
    @executeEpisodeEMG, false, 0, false, [14, 14]);
    %} 


    addpath(genpath(path_to_framework));
    addpath(genpath('utils'));
    addpath(genpath('LabEPN'));
    addpath('RLSetup');
    addpath('Experiments');

    prepare_environment(path_to_data, verbose_level-1);


    % the directory where the model results will be put
    dir_model = 'Experiments/results/'+model_name;

    % creates the directory if it doesnt exists
    [~, ~, ~] = mkdir(dir_model);

    % we can run a lot of experiments
    number_experiments = numel(experiment_ids);

    % this reads all the parameter table for selecting only rows in future
    parameters = readtable('Experiments/experiments_parameters_QNN.csv');


    % Set general parameters
    context = containers.Map();
    % 1 = Each user with diferent model
    % 2 = Serial-user together in same model
    context('tabulation_mode') = 2;
    context('is_preprocessed') = is_preprocessed;
    context('noGestureDetection') = noGestureDetection;

    if verbose_level > 0
        fprintf("Generating orientation...\n");
    end
    assignin('base','packetEMG',     false); 
    
    rotation_generated = false;
    
    filename_model = "model_" + model_name;
        
    excel_dir = dir_model + "/results.xlsx";
    
    

    t_begin = tic;
    
    sheet_name = "General";
    
    for index_experiment_ids=1:number_experiments
        
        % just get the experiment id from csv table
        experiment_id = experiment_ids(index_experiment_ids);
        % only get the row with the set of params for experiment
        params_experiment_row = parameters(experiment_id, :);


        if verbose_level > 0
            fprintf("Setting hyper parameters and models for experiment %d\n", experiment_id);
        end
        [params, nnConfig, qLearningConfig] = build_params(params_experiment_row, context);
        
        
        if ~rotation_generated && ~is_preprocessed
            on = true;
            off = false;
            environment_options = struct();
            environment_options.post_processing = on;
            environment_options.randomGestures = off;
            environment_options.noGestureDetection = noGestureDetection;
            context('noGestureDetection') = environment_options.noGestureDetection;
            environment_options.rangeValues = 150;
            environment_options.packetEMG = true;
            prepare_environment(path_to_data, verbose_level-1, environment_options);

            Code_0(1, path_to_data);  % params.rangeDownTrain
            orientation      = evalin('base', 'orientation');
            dataPacket = evalin('base','dataPacket');
            if verbose_level > 0
                fprintf("Orientation generated\n");
            end
            context('orientation') = orientation;
            context('dataPacket') = dataPacket;
            rotation_generated = true;
        end
        
        % for each user
        if verbose_level > 0
            fprintf("*****Training with %d users, each one with %d gestures*****\n", params.num_users, params.RepTraining);
        end
        
        % init model once for all users and not reset
        qLearningConfig.total_episodes = params.RepTraining * params.num_users;
        qLearningConfig.total_episodes_test = params.RepTesting * params.num_users_test;
        q_neural_network = QNeuralNetwork(params.sequential_conv_network, params.sequential_network, ...
                    nnConfig, qLearningConfig, executeEpisodeEMG);    % @executeEpisodeEMGImage 

        q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG); 
        
        
        
                
        
        for g_epoch=1:params.global_epochs
            % just final epoch
            shape_full_history = [1, params.RepTraining];
            history_classification_window_correct = zeros(shape_full_history);
            history_classification_window_incorrect = zeros(shape_full_history);
            history_classification_class_correct = zeros(shape_full_history);
            history_classification_class_incorrect = zeros(shape_full_history);
            history_classification_recognition_correct = zeros(shape_full_history);
            history_classification_recognition_incorrect = zeros(shape_full_history);

            history_rewards_train = [];
            history_update_costs_train = {};
        
            for index_id_user=1:params.num_users
                % extracting user vars

                user_real_id = params.list_users(index_id_user);

                user_folder = "user"+user_real_id;
                if verbose_level > 0
                    fprintf("Experiment: %d/%d, %s: %d/%d\n", index_experiment_ids, ...
                        number_experiments, user_folder, index_id_user, params.num_users);
                end


                if is_preprocessed
                    % just use the feature table in the path to data
                    userData = loadUserByNameAndDir(user_folder, path_to_data, false);
                    context('user_gestures') = userData.training(randperm(numel(userData.training)));

                else
                    % use LabEPN Code 1 for fetching data
                    userData = loadSpecificUserByName(user_folder, path_to_data);
                    index_in_packet = getUserIndexInPacket(dataPacket, user_folder);
                    assignin('base', 'userIndex', index_in_packet);
                    assignin('base','index_user', index_in_packet-2);
                    assignin('base','rangeDown', params.rangeDownTrain);
                    assignin('base','emgRepetition', params.rangeDownTrain);
                    energy_index = strcmp(orientation(:,1), userData.userInfo.name);
                    rand_data=orientation{energy_index,6};
                    context('rand_data') = rand_data; 

                    context('user_gestures') = userData.training;

                end


                row = (1 + initial_row_position + index_experiment_ids);


                context('offset_user') = (index_id_user-1) * params.RepTraining;

                history_episodes_train = q_neural_network.runEpisodes(@getRewardEMG, false, context, verbose_level-1);

                history_classification_window_correct = history_classification_window_correct + history_episodes_train('history_classification_window_correct');
                history_classification_window_incorrect = history_classification_window_incorrect + history_episodes_train('history_classification_window_incorrect');
                history_classification_class_correct = history_classification_class_correct + history_episodes_train('history_classification_class_correct');
                history_classification_class_incorrect = history_classification_class_incorrect + history_episodes_train('history_classification_class_incorrect');
                history_classification_recognition_correct = history_classification_recognition_correct + history_episodes_train('history_classification_recognition_correct');
                history_classification_recognition_incorrect = history_classification_recognition_incorrect + history_episodes_train('history_classification_recognition_incorrect');

                history_rewards = history_episodes_train('history_rewards');
                history_update_costs = history_episodes_train('history_update_costs');
                history_rewards_train = [history_rewards_train, history_rewards];
                history_update_costs_train = [history_update_costs_train, history_update_costs];


            end
            
            % q_neural_network.nnConfig.learning_rate = q_neural_network.alpha;
        end
            
        
        
        
        
        % testing
        shape_full_history_test = [1, params.RepTesting];
        history_classification_window_correct_test = zeros(shape_full_history_test);
        history_classification_window_incorrect_test = zeros(shape_full_history_test);
        history_classification_class_correct_test = zeros(shape_full_history_test);
        history_classification_class_incorrect_test = zeros(shape_full_history_test);
        history_classification_recognition_correct_test = zeros(shape_full_history_test);
        history_classification_recognition_incorrect_test = zeros(shape_full_history_test);
        
        history_rewards_test = [];
        
        for index_id_user=1:params.num_users
            % extracting user vars
            
            user_real_id = params.list_users(index_id_user);

            user_folder = "user"+user_real_id;
            if verbose_level > 0
                fprintf("Testing experiment: %d/%d, %s: %d/%d\n", index_experiment_ids, ...
                    number_experiments, user_folder, index_id_user, params.num_users);
            end
            
            
            if is_preprocessed
                % just use the feature table in the path to data
                userData = loadUserByNameAndDir(user_folder, path_to_data, false);
                context('user_gestures') = userData.training(randperm(numel(userData.training)));
        
            else
                % use LabEPN Code 1 for fetching data
                userData = loadSpecificUserByName(user_folder, path_to_data);
                index_in_packet = getUserIndexInPacket(dataPacket, user_folder);
                assignin('base', 'userIndex', index_in_packet);
                assignin('base','index_user', index_in_packet-2);
                assignin('base','rangeDown', params.rangeDownTrain);
                assignin('base','emgRepetition', params.rangeDownTrain);
                energy_index = strcmp(orientation(:,1), userData.userInfo.name);
                rand_data=orientation{energy_index,6};
                context('rand_data') = rand_data; 
                
                context('user_gestures') = userData.training(rand_data(:), 1);
        
            end
            
            
            context('offset_user') = (index_id_user-1) * params.RepTesting;
           

            if is_preprocessed
                context('user_gestures') = userData.testing(randperm(numel(userData.testing)));
            else
                context('user_gestures') = userData.testing;
            end


            history_episodes_test = q_neural_network.runEpisodes(@getRewardEMG, true, context, verbose_level-1);

            history_classification_window_correct_test = history_classification_window_correct_test + history_episodes_test('history_classification_window_correct');
            history_classification_window_incorrect_test = history_classification_window_incorrect_test + history_episodes_test('history_classification_window_incorrect');
            history_classification_class_correct_test = history_classification_class_correct_test + history_episodes_test('history_classification_class_correct');
            history_classification_class_incorrect_test = history_classification_class_incorrect_test + history_episodes_test('history_classification_class_incorrect');
            history_classification_recognition_correct_test = history_classification_recognition_correct_test + history_episodes_test('history_classification_recognition_correct');
            history_classification_recognition_incorrect_test = history_classification_recognition_incorrect_test + history_episodes_test('history_classification_recognition_incorrect');
                     
            history_rewards = history_episodes_test('history_rewards');
            history_rewards_test = [history_rewards_test, history_rewards];
            
            
            
        end
        
        
        row = (1 + initial_row_position + index_experiment_ids);

        history_episodes_train_full = containers.Map();
        history_episodes_train_full('history_classification_window_correct') = history_classification_window_correct;
        history_episodes_train_full('history_classification_window_incorrect') = history_classification_window_incorrect;
        history_episodes_train_full('history_classification_class_correct') = history_classification_class_correct;
        history_episodes_train_full('history_classification_class_incorrect') = history_classification_class_incorrect;
        history_episodes_train_full('history_classification_recognition_correct') = history_classification_recognition_correct;
        history_episodes_train_full('history_classification_recognition_incorrect') = history_classification_recognition_incorrect;
        history_episodes_train_full('history_rewards') = history_rewards_train;
        history_episodes_train_full('history_update_costs') = history_update_costs_train;
        
            
            
        history_episodes_test_full = containers.Map();
        history_episodes_test_full('history_classification_window_correct') = history_classification_window_correct_test;
        history_episodes_test_full('history_classification_window_incorrect') = history_classification_window_incorrect_test;
        history_episodes_test_full('history_classification_class_correct') = history_classification_class_correct_test;
        history_episodes_test_full('history_classification_class_incorrect') = history_classification_class_incorrect_test;
        history_episodes_test_full('history_classification_recognition_correct') = history_classification_recognition_correct_test;
        history_episodes_test_full('history_classification_recognition_incorrect') = history_classification_recognition_incorrect_test;
        history_episodes_test_full('history_rewards') = history_rewards_test;
        
        
        
         % % results
        if verbose_level > 0
            fprintf("***Results***\n");
        end
        
        users_string = "";
        for index_list_users=1:length(params.list_users)
            users_string = users_string + params.list_users(index_list_users) + "-";
        end
        
        [history_experiments, index_history_experiment] = Experiment.getTableResults(history_episodes_train_full, ...
            history_episodes_test_full, users_string, experiment_id, verbose_level-1);


        preheader = {"", "", "Training accuracy", "", "", "Testing accuracy", "", "", "Reward"};
        writetable(cell2table(preheader), excel_dir, 'Sheet',sheet_name,'Range',"A1",'WriteVariableNames',false);
        header = {"USER", "Experiment", "Window", "Classification", "Recognition", "Window", "Classification", "Recognition", "Mean reward"};
        writetable(cell2table(header), excel_dir, 'Sheet',sheet_name,'Range',"A2",'WriteVariableNames',false);

            
        writetable(cell2table(history_experiments(1, 1:index_history_experiment-1)), excel_dir,'Sheet', "General",'Range',"A" + row,'WriteVariableNames', false);

        saveas_name = dir_model + "/Figure-" + filename_model + "-Experiment_" + experiment_id +"_reward_and_cost.png";
        Experiment.plotAndSave(history_episodes_train_full, saveas_name);

        % save only one model per experiment
        model_dir = dir_model + "/model_general_experiment_" + experiment_id + ".mat";
        save(model_dir,'params', 'q_neural_network');

        

    end
    t_end = toc(t_begin);
    if verbose_level > 0
        fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
    end


end