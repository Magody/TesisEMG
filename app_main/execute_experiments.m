function execute_experiments(model_name, experiment_ids, ...
    path_to_framework, path_to_data, verbose_level)

    % execute_experiments("test", [1, 0], "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework", '/home/magody/programming/MATLAB/tesis/Data/', 3);


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
    context('data_dir') = path_to_data;
    
    
    processed_orientation = false;
    
    
    filename_model = "model_" + model_name;


    
    excel_dir = dir_model + "/results.xlsx";


    for index_experiment_ids=1:number_experiments

        % just get the experiment id from csv table
        experiment_id = experiment_ids(index_experiment_ids);
        % only get the row with the set of params for experiment
        params_experiment_row = parameters(experiment_id+1, :);


        if verbose_level > 0
            fprintf("Setting hyper parameters and models for experiment %d\n", experiment_id);
        end
        [params, nnConfig, qLearningConfig] = build_params(params_experiment_row, context);
        
        generate_rng(params.seed_rng);
        
        
        
        if ~processed_orientation
            if verbose_level > 0
                fprintf("Generating orientation...\n");
            end
            assignin('base','packetEMG',     false); 
            Code_0(params.rangeDownTrain, path_to_data);  % rangeDown
            orientation      = evalin('base', 'orientation');
            dataPacket = evalin('base','dataPacket');
            if verbose_level > 0
                fprintf("Orientation generated\n");
            end
            context('orientation') = orientation;
            context('dataPacket') = dataPacket;
            processed_orientation = true;
        end
        
        
        generate_rng(params.seed_rng);

        q_neural_network = QNeuralNetwork(params.sequential_conv_network, params.sequential_network, ...
                        nnConfig, qLearningConfig, @executeEpisodeEMG);    % @executeEpisodeEMGImage 

        q_neural_network.setCustomRunEpisodes(@customRunEpisodesEMG);

        if verbose_level > 0
            fprintf("*****Training with %d users, each one with %d gestures*****\n", params.num_users, params.RepTraining);
        end
        t_begin = tic;
        history_episodes_train = q_neural_network.runEpisodes(@getRewardEMG, false, context, verbose_level-1);
        t_end = toc(t_begin);
        if verbose_level > 0
            fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);
        end

        % % plot results
        % disp(history_episodes_train('history_gestures_name'));
        figure(1);
        subplot(1,2,1)
        history_rewards = history_episodes_train('history_rewards')';
        plot(history_rewards(:));
        title("Train: Reward");

        subplot(1,2,2)
        linear_update_costs = [];
        update_costs_by_episode = history_episodes_train('history_update_costs');

        for index_user=1:size(update_costs_by_episode, 1)
            for index_gesture=1:size(update_costs_by_episode, 2)
                costs = update_costs_by_episode{index_user, index_gesture};
                linear_update_costs = [linear_update_costs; costs(:)];
            end
        end
        plot(linear_update_costs);
        title("Cost");

        saveas(gcf, dir_model + "/Figure_" + filename_model + "-Experiment_" + experiment_id +"_reward_and_cost.png");

        close all;


        train_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_window_correct'), history_episodes_train('history_classification_window_incorrect'));
        train_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_class_correct'), history_episodes_train('history_classification_class_incorrect'));

        if verbose_level > 0
            fprintf("Train: Mean accuracy for classification window: %.4f\n", train_metrics_classification_window('accuracy'));
            fprintf("Train: Mean accuracy for classification class: %.4f\n", train_metrics_classification_class('accuracy'));
        end

        % % test
        if verbose_level > 0
            fprintf("*****Test with %d users, each one with %d gestures*****\n", params.num_users_test, params.RepTesting);
        end

        history_episodes_test = q_neural_network.runEpisodes(@getRewardEMG, true, context, verbose_level-1);

        test_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_window_correct'), history_episodes_test('history_classification_window_incorrect'));
        test_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_class_correct'), history_episodes_test('history_classification_class_incorrect'));

        if verbose_level > 0
            fprintf("Test: Mean collected reward %.4f\n", mean(history_episodes_test("history_rewards")));
            fprintf("Test: Mean accuracy for classification window: %.4f\n", test_metrics_classification_window('accuracy'));
            fprintf("Test: Mean accuracy for classification class: %.4f\n", test_metrics_classification_class('accuracy'));
        end



        history_experiments = cell([1, 20]);
        % id experiment
        history_experiments{1, 1} = experiment_id;
        
        
        
        
        for index_user=1:size(update_costs_by_episode, 1)
            index_history_experiment = 2;
            % execution time
            % history_experiments{index_experiment_ids, index_history_experiment} = t_end/60;
            % index_history_experiment = index_history_experiment + 1;
            
            
            list_users_train = context('list_users');
            % Train: accuracy window
            train_metrics_classification_window_by_user = train_metrics_classification_window('accuracy_by_user');
            history_experiments{1, index_history_experiment} = train_metrics_classification_window_by_user(index_user);
            index_history_experiment = index_history_experiment + 1;
            % Train: accuracy class
            train_metrics_classification_class_by_user = train_metrics_classification_class('accuracy_by_user');
            history_experiments{1, index_history_experiment} = train_metrics_classification_class_by_user(index_user);
            index_history_experiment = index_history_experiment + 1;
            
            
            % Test: accuracy class
            test_metrics_classification_window_by_user = test_metrics_classification_window('accuracy_by_user');
            history_experiments{1, index_history_experiment} = test_metrics_classification_window_by_user(index_user);
            index_history_experiment = index_history_experiment + 1;
            % Test: accuracy class
            test_metrics_classification_class_by_user = test_metrics_classification_class('accuracy_by_user');
            history_experiments{1, index_history_experiment} = test_metrics_classification_class_by_user(index_user);
            index_history_experiment = index_history_experiment + 1;
            
            
            user_id = list_users_train(index_user);
            sheet_name = "USER"+user_id;
            row = (1 + index_experiment_ids);
            
            
            % writes a Sheet with header and all params used
            writetable(cell2table(history_experiments(1, 1:index_history_experiment-1)), excel_dir,'Sheet', sheet_name,'Range',"A" + row,'WriteVariableNames', false);

        end
        
        model_dir = dir_model + "/model_experiment_" + experiment_id + ".mat";
        save(model_dir,'params', 'q_neural_network');
        


        

    end


end