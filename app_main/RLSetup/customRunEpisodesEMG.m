function history_episodes = customRunEpisodesEMG(q_neural_network, functionGetReward, is_test, context, verbose_level)
    % each user has their episodes

    window_size = context('window_size');
    stride = context('stride');
    
    if is_test
        totalGestures = context('RepTesting');
        rangeDown = context('rangeDownTest');
        user_gestures = context('user_gestures');
    else
        totalGestures = context('RepTraining');
        rangeDown = context('rangeDownTrain');
        user_gestures = context('user_gestures');
    end
    
    tabulation_mode = context('tabulation_mode');
    offset_user = context('offset_user');
    
    
    history_episodes = containers.Map();
    shape_history = [1, totalGestures];
    
    history_rewards = zeros(shape_history);
    history_update_costs = cell(shape_history);
    
    history_classification_window_correct = zeros(shape_history);
    history_classification_window_incorrect = zeros(shape_history);
    
    history_classification_class_correct = zeros(shape_history);
    history_classification_class_incorrect = zeros(shape_history);
    
    history_classification_recognition_correct = zeros(shape_history);
    history_classification_recognition_incorrect = zeros(shape_history);
    
    history_gestures_name = cell(shape_history);
    
    index_id_user = 1;
    
    is_preprocessed = context('is_preprocessed');
    
    gesture_counter = 1;
    
    if ~is_preprocessed
       rand_data = context('rand_data'); 
    end
    
    for gesture_number=rangeDown:rangeDown+totalGestures-1

        if is_preprocessed
            user_gesture_struct = user_gestures{gesture_counter};
        else
            user_gesture_struct = user_gestures{rand_data(gesture_number), 1};
        end
        
        
        
                
        
        
        
        context('gestureName') = string(user_gesture_struct.gestureName);
        
        if is_preprocessed
            key_features = "features_per_window" + "Win" + window_size + "Stride" + stride;
            context('features_per_window') = user_gesture_struct.(key_features);
            context('num_windows') = size(context('features_per_window'), 1);  % num_windows x 40
            context('emg_points') = size(user_gesture_struct.emg, 1);
        else
            context('emg') = user_gesture_struct.emg;
        end
        

        if context('gestureName') == "noGesture"
             context('ground_truth_index') = [0, 0];
        else
             context('ground_truth_index') = user_gesture_struct.groundTruthIndex;
        end

        if tabulation_mode == 1
            % each user is a world
            episode = gesture_counter;
        elseif tabulation_mode == 2
            episode = offset_user + gesture_counter;
        end

        history_episode = q_neural_network.functionExecuteEpisode(q_neural_network, episode, is_test, functionGetReward, context, verbose_level-1);



        history_gestures_name{index_id_user, gesture_counter} = context('gestureName');
        history_rewards(index_id_user, gesture_counter) = history_episode('reward_cummulated');


        update_costs = history_episode('update_costs');
        history_update_costs{index_id_user, gesture_counter} = update_costs(:);

        history_classification_window_correct(index_id_user, gesture_counter) = history_episode('classification_window_correct');
        history_classification_window_incorrect(index_id_user, gesture_counter) = history_episode('classification_window_incorrect');

        history_classification_class_correct(index_id_user, gesture_counter) = history_episode('classification_class_correct');
        history_classification_class_incorrect(index_id_user, gesture_counter) = history_episode('classification_class_incorrect');

        history_classification_recognition_correct(index_id_user, gesture_counter) = history_episode('classification_recognition_correct');
        history_classification_recognition_incorrect(index_id_user, gesture_counter) = history_episode('classification_recognition_incorrect');


        if ~is_test
            % QNN target strategy, for "stable" learning
            q_neural_network.updateQNeuralNetworkTarget();
        end

        gesture_counter = gesture_counter + 1;

    end
    
    history_episodes('history_rewards') = history_rewards;
    history_episodes('history_update_costs') = history_update_costs;
    
    
    history_episodes('history_classification_window_correct') = history_classification_window_correct;
    history_episodes('history_classification_window_incorrect') = history_classification_window_incorrect;
    
    history_episodes('history_classification_class_correct') = history_classification_class_correct;
    history_episodes('history_classification_class_incorrect') = history_classification_class_incorrect;
    
    history_episodes('history_classification_recognition_correct') = history_classification_recognition_correct;
    history_episodes('history_classification_recognition_incorrect') = history_classification_recognition_incorrect;
    
    
    history_episodes('history_gestures_name') = history_gestures_name;
    
    
            
end

