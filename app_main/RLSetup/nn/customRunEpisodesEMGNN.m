function history_episodes = customRunEpisodesEMGNN(q_neural_network, functionGetReward, is_test, context, verbose_level)
    
    data_dir = context('data_dir');
    window_size = context('window_size');
    stride = context('stride');
    
    if is_test
        totalGestures = context('RepTesting');
        rangeDown = context('rangeDownTest');
        num_users = context('num_users_test');
        list_users = context('list_users_test');
        total_episodes = q_neural_network.qLearningConfig.total_episodes_test;
    else
        totalGestures = context('RepTraining');
        rangeDown = context('rangeDownTrain');
        num_users = context('num_users');
        list_users = context('list_users');
        total_episodes = q_neural_network.qLearningConfig.total_episodes;
    end
    
    
    history_episodes = containers.Map();
    shape_history = [num_users, totalGestures];
    
    history_rewards = zeros(shape_history);
    history_update_costs = cell(shape_history);
    
    history_classification_window_correct = zeros(shape_history);
    history_classification_window_incorrect = zeros(shape_history);
    
    history_classification_class_correct = zeros(shape_history);
    history_classification_class_incorrect = zeros(shape_history);
    
    history_gestures_name = cell(shape_history);
    
    
    step_print = totalGestures/5;
    
    
    for index_id_user=1:num_users
        % extracting user vars
        
        user_folder = "user"+list_users(index_id_user);
        userData = loadUserByNameAndDir(user_folder, data_dir, false);
        
        if is_test
            user_gestures_part1 = userData.testing;
        else
            user_gestures_part1 = userData.training;
        end
        
        offset_user = (index_id_user - 1) * totalGestures;
        
        gesture_counter = 1;
        for gesture_number=rangeDown:rangeDown+totalGestures-1
            
            if mod(gesture_number, step_print) == 0 && ~is_test
                fprintf("| user: %s, %d/%d, gesture: %d/%d\n", userData.userInfo.name, index_id_user, num_users, gesture_number, totalGestures);
            end
            
            user_gesture_struct = user_gestures_part1{gesture_counter};
            
            key_features = "features_per_window" + "Win" + window_size + "Stride" + stride;
            context('features_per_window') = user_gesture_struct.(key_features);
            context('num_windows') = size(context('features_per_window'), 1);  % num_windows x 40
            context('gestureName') = string(user_gesture_struct.gestureName);
            
            if context('gestureName') == "noGesture"
                 context('ground_truth_index') = [0, 0];
            else
                 context('ground_truth_index') = user_gesture_struct.groundTruthIndex;
            end
            
            
            % linear_index = offset_user + gesture_number;
            
            
            episode = offset_user + gesture_counter;
            history_episode = q_neural_network.functionExecuteEpisode(q_neural_network, episode, is_test, functionGetReward, context, verbose_level-1);
            
            history_gestures_name{index_id_user, gesture_counter} = context('gestureName');
            history_rewards(index_id_user, gesture_counter) = history_episode('reward_cummulated');

            
            update_costs = history_episode('update_costs');
            history_update_costs{index_id_user, gesture_counter} = update_costs(:);
            
            history_classification_window_correct(index_id_user, gesture_counter) = history_episode('classification_window_correct');
            history_classification_window_incorrect(index_id_user, gesture_counter) = history_episode('classification_window_incorrect');
 
            history_classification_class_correct(index_id_user, gesture_counter) = history_episode('classification_class_correct');
            history_classification_class_incorrect(index_id_user, gesture_counter) = history_episode('classification_class_incorrect');
            
            
    

            if ~is_test
                % QNN target strategy, for "stable" learning
                q_neural_network.updateQNeuralNetworkTarget();
            end
            
            gesture_counter = gesture_counter + 1;
            
        end
        
    end
    
    history_episodes('history_rewards') = history_rewards;
    history_episodes('history_update_costs') = history_update_costs;
    
    
    history_episodes('history_classification_window_correct') = history_classification_window_correct;
    history_episodes('history_classification_window_incorrect') = history_classification_window_incorrect;
    
    history_episodes('history_classification_class_correct') = history_classification_class_correct;
    history_episodes('history_classification_class_incorrect') = history_classification_class_incorrect;
    
    history_episodes('history_gestures_name') = history_gestures_name;
    
    
            
end

