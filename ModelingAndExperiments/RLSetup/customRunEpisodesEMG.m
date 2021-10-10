function history_episodes = customRunEpisodesEMG(q_neural_network, functionGetReward, type_execution, context, verbose_level)
    % each user has their episodes

    t = 1;
    
    window_size = context('window_size');
    stride = context('stride');
    
    is_validation_or_test = type_execution ~= 1;
    is_train_only = type_execution == 1;
    is_validation_only = type_execution == 2;
    is_test_only = type_execution == 3;
    
    if is_train_only
        % Train
        t = context('global_epoch');
        user_gestures = context('user_gestures_train');
        rangeDown = context('rangeDownTrain');
    elseif is_validation_only
        % Validation
        user_gestures = context('user_gestures_validation');
        rangeDown = context('rangeDownValidation');
    elseif is_test_only
        % Test
        rangeDown = context('rangeDownTest');
        user_gestures = context('user_gestures_test');        
    end
    
    totalGestures = length(user_gestures);
    
    % tabulation_mode = context('tabulation_mode');
    % offset_user = context('offset_user');
    
    
    history_episodes = struct();
    shape_history = [1, totalGestures];
    
    history_rewards = zeros(shape_history);
    history_update_costs = cell(shape_history);
    
    history_classification_window_correct = zeros(shape_history);
    history_classification_window_incorrect = zeros(shape_history);
    
    history_classification_correct = zeros(shape_history);
    history_classification_incorrect = zeros(shape_history);
    
    history_recognition_correct = zeros(shape_history);
    history_recognition_incorrect = zeros(shape_history);
    
    history_responses = cell(shape_history);
    
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
        
       
        if is_preprocessed
            key_features = "features_per_window" + "Win" + window_size + "Stride" + stride;
            context('features_per_window') = user_gesture_struct.(key_features);
            context('num_windows') = size(context('features_per_window'), 1);  % num_windows x 40
            context('emg_points') = size(user_gesture_struct.emg, 1);
        else
            context('emg') = user_gesture_struct.emg;
        end
        
        if is_train_only || is_validation_only
            context('gestureName') = string(user_gesture_struct.gestureName);


            if context('gestureName') == "noGesture"
                 context('ground_truth_index') = [0, 0];
            else
                 context('ground_truth_index') = user_gesture_struct.groundTruthIndex;
            end
        end


        history_episode = q_neural_network.functionExecuteEpisode(q_neural_network, t, type_execution, functionGetReward, context, verbose_level-1);

        history_rewards(index_id_user, gesture_counter) = history_episode('reward_cummulated');

        history_responses{index_id_user, gesture_counter} = history_episode('response');

        if is_train_only
            update_costs = history_episode('update_costs');
            history_update_costs{index_id_user, gesture_counter} = update_costs(:);
            % QNN target strategy, for "stable" learning
            q_neural_network.updateQNeuralNetworkTarget();
        end
        
        if is_train_only || is_validation_only

            history_classification_window_correct(index_id_user, gesture_counter) = history_episode('classification_window_correct');
            history_classification_window_incorrect(index_id_user, gesture_counter) = history_episode('classification_window_incorrect');

            history_classification_correct(index_id_user, gesture_counter) = history_episode('classification_class_correct');
            history_classification_incorrect(index_id_user, gesture_counter) = history_episode('classification_class_incorrect');

            history_recognition_correct(index_id_user, gesture_counter) = history_episode('classification_recognition_correct');
            history_recognition_incorrect(index_id_user, gesture_counter) = history_episode('classification_recognition_incorrect');

        end
            
        gesture_counter = gesture_counter + 1;

    end
    
    history_episodes.history_rewards = history_rewards;
    history_episodes.history_responses = history_responses;
    
    if is_train_only
        history_episodes.history_update_costs = history_update_costs;
    end
    
    if is_train_only || is_validation_only
    
        history_episodes.history_classification_window_correct = history_classification_window_correct;
        history_episodes.history_classification_window_incorrect = history_classification_window_incorrect;

        history_episodes.history_classification_correct = history_classification_correct;
        history_episodes.history_classification_incorrect = history_classification_incorrect;

        history_episodes.history_recognition_correct = history_recognition_correct;
        history_episodes.history_recognition_incorrect = history_recognition_incorrect;
    end
    
    
    
            
end

