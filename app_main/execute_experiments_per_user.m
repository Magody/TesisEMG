function execute_experiments_per_user(model_name, ...
    path_to_framework, path_to_data, verbose_level, ...
    executeEpisodeEMG, noGestureDetection, initial_row_position, is_preprocessed, ...
    experiment_ids)

    %{
    execute_experiments_per_user("test_cnn_experiments", ...
    "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework", ...
    '/home/magody/programming/MATLAB/tesis/Data/preprocessing/', 2, ...
    @executeEpisodeEMG, false, 0, true, [14, 15]);
    
    execute_experiments_per_user("test_cnn_experiments", ...
    "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework", ...
    '/home/magody/programming/MATLAB/tesis/Data/', 2, ...
    @executeEpisodeEMG, false, 0, false, [0, 0]);
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
    context('tabulation_mode') = 1;
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
    
    for index_experiment_ids=1:number_experiments
        
        % just get the experiment id from csv table
        experiment_id = experiment_ids(index_experiment_ids);
        % only get the row with the set of params for experiment
        params_experiment_row = parameters(experiment_id+1, :);


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
            
            
            row = (2 + initial_row_position + index_experiment_ids);
        
            % each user individually
            % reset model on new user
            [params, nnConfig, qLearningConfig] = build_params(params_experiment_row, context);

            qLearningConfig.total_episodes = params.RepTraining;
            qLearningConfig.total_episodes_test = params.RepTesting;
            q_neural_network = QNeuralNetwork(params.sequential_conv_network, params.sequential_network, ...
                    nnConfig, qLearningConfig, executeEpisodeEMG);    % @executeEpisodeEMGImage 

            q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG);  
            
            
            context('offset_user') = (index_id_user-1) * params.RepTraining;
        
            for g_epoch=1:params.global_epochs
                history_episodes_train = q_neural_network.runEpisodes(@getRewardEMG, false, context, verbose_level-1);
                q_neural_network.epsilon = q_neural_network.qLearningConfig.initial_epsilon;
                % q_neural_network.nnConfig.learning_rate = q_neural_network.alpha;
            end
            
            sheet_name = "USER"+user_real_id;

            % test with data from same user
             % % test
            if verbose_level > 0
                fprintf("Test for %s, using %d gestures\n", user_folder, params.num_users_test);
            end

            if is_preprocessed
                context('user_gestures') = userData.testing(randperm(numel(userData.testing)));
            else
                context('user_gestures') = userData.testing;
            end


            history_episodes_test = q_neural_network.runEpisodes(@getRewardEMG, true, context, verbose_level-1);

             % % results
            if verbose_level > 0
                fprintf("***Results***\n");
            end
            [history_experiments, index_history_experiment] = Experiment.getTableResults(history_episodes_train, ...
                history_episodes_test, user_real_id, experiment_id, verbose_level-1);

            preheader = {"", "", "Training accuracy", "", "", "Testing accuracy", "", "", "Reward"};
            writetable(cell2table(preheader), excel_dir, 'Sheet',sheet_name,'Range',"A1",'WriteVariableNames',false);
            header = {"USER", "Experiment", "Window", "Classification", "Recognition", "Window", "Classification", "Recognition", "Mean reward"};
            writetable(cell2table(header), excel_dir, 'Sheet',sheet_name,'Range',"A2",'WriteVariableNames',false);

            writetable(cell2table(history_experiments(1, 1:index_history_experiment-1)), excel_dir,'Sheet', sheet_name,'Range',"A" + row,'WriteVariableNames', false);

            saveas_name = dir_model + "/Figure-" + sheet_name + "-" + filename_model + "-Experiment_" + experiment_id +"_reward_and_cost.png";
            Experiment.plotAndSave(history_episodes_train, saveas_name);

            % save each model in apart way
            model_dir = dir_model + "/model_" + sheet_name + "_experiment_" + experiment_id + ".mat";
            save(model_dir,'params', 'q_neural_network');
       
        
        end
        
        
    end
    t_end = toc(t_begin);
    if verbose_level > 0
        fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
    end


end