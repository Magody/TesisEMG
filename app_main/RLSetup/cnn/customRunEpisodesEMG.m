function history_episodes = customRunEpisodesEMG(q_neural_network, functionGetReward, is_test, context, verbose_level)
    
    orientation = context('orientation');
    
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
    
    
    dataPacket = context('dataPacket');
    
    
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
        userData = loadSpecificUserByName(user_folder, context('data_dir'));
        
        
        index_in_packet = getUserIndexInPacket(dataPacket, user_folder);
        assignin('base', 'userIndex', index_in_packet);
        assignin('base','index_user', index_in_packet-2);
        assignin('base','rangeDown', rangeDown);
        assignin('base','emgRepetition', rangeDown);

        energy_index = strcmp(orientation(:,1), userData.userInfo.name);
        rand_data=orientation{energy_index,6};
        
        context('rand_data') = rand_data;


        offset_user = (index_id_user - 1) * totalGestures;
        
        classification_class_correct_user = 0;
        classification_class_incorrect_user = 0;
        
        gesture_counter = 1;
        for gesture_number=rangeDown:rangeDown+totalGestures-1
            
            if mod(gesture_number, step_print) == 0 && ~is_test
                fprintf("| user: %s, %d/%d, gesture: %d/%d\n", userData.userInfo.name, index_id_user, num_users, gesture_number, totalGestures);
            end
                

            ground_truth_Index = [0, 0];
            
            if gesture_number > 150
                gestureName = userData.testing{rand_data(gesture_number-150),1}.gestureName;
                if gestureName ~= "noGesture"
                    ground_truth_Index = userData.testing{rand_data(gesture_number),1}.groundTruthIndex;
                end
                emg = userData.testing{rand_data(gesture_number-150),1}.emg;
            else
                gestureName = userData.training{rand_data(gesture_number),1}.gestureName;
                if gestureName ~= "noGesture"
                    ground_truth_Index = userData.training{rand_data(gesture_number),1}.groundTruthIndex;
                end
                emg = userData.training{rand_data(gesture_number),1}.emg;
            end
            
            context('gestureName') = gestureName;
            context('ground_truth_Index') = ground_truth_Index;
            context('emg') = emg;
            
            % linear_index = offset_user + gesture_number;
            
            
            episode = offset_user + gesture_counter;
            history_episode = q_neural_network.functionExecuteEpisode(q_neural_network, episode, is_test, functionGetReward, context, verbose_level-1);
            
            history_gestures_name{index_id_user, gesture_counter} = string(gestureName);
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

