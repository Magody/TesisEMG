function history_episode = executeEpisodeEMGNN(q_neural_network, episode, is_test, functionGetReward, context, verbose_level) 


    
    window_size = context('window_size');
    stride = context('stride');
    gestureName = context('gestureName');
    ground_truth_Index = context('ground_truth_index');
    features_per_window = context('features_per_window');
    num_windows = context('num_windows');
    interval_for_learning = context('interval_for_learning'); 
    
     classes = containers.Map(["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"], {1, 2, 3, 4, 5, 6});
    real_class = classes(string(gestureName));
    predictions = [];
    
    index_state_begin = 1;
    index_state_end = window_size;

    gt_gestures_pts=zeros([1,num_windows]);
    gt_gestures_pts(1,1)=window_size;

    for k = 1:num_windows-1
        gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
    end
    
    gt_gestures_labels = mapGroundTruthToLabelsWithPts(gt_gestures_pts, ground_truth_Index, gestureName, 0.2);

    gt_gestures_labels_num = mapGestureLabelsToNumbers(num_windows, gt_gestures_labels);


    
    history_episode = containers.Map();
    update_counter = 0;
    update_costs = [];
    reward_cummulated = 0;
    
    classification_window_correct = 0;
    classification_window_incorrect = 0;
    
    
    step_t = 0;
    t_for_learning = interval_for_learning; % floor(num_windows/interval_for_learning);
    
    for window=1:num_windows-1
        
        step_t = step_t + 1;


        % slice = emg(:, index_state_begin:index_state_end, :);slice(:);
        state = features_per_window(window, :)';

        [Qval, action] = q_neural_network.selectAction(state, is_test); %#ok<ASGLU>
        
        if action ~= 6 || real_class == 6
            % if real class is noGesture, we must save the predictions
            predictions = [predictions, action];            
        end

        context('real_action') = gt_gestures_labels_num(window+1);
        % context is modified by reference
        [reward, new_state, finish] = functionGetReward(state, action, context);
        % in this case, new_state is useless
        finish = window == num_windows-1;
        
        if reward > 0
            classification_window_correct = classification_window_correct + 1;
        else
            classification_window_incorrect = classification_window_incorrect + 1;
        end
        
        if ~is_test
            q_neural_network.saveExperienceReplay(state, action, reward, new_state, finish);
            
            if mod(step_t, t_for_learning) == 0
                % update in each step could be very brute
                history_learning = q_neural_network.learnFromExperienceReplay(episode, verbose_level);
                if history_learning('learned')
                    update_counter = update_counter + 1;
                    update_costs(1, update_counter) = history_learning('mean_cost');
                end
            end
            
        end

        reward_cummulated = reward_cummulated + reward;

        index_state_begin = index_state_begin + stride;
        index_state_end = index_state_end + stride;

    end
    
    
    

    history_episode('reward_cummulated') = reward_cummulated;
    history_episode('update_costs') = update_costs;
    
    history_episode('classification_window_correct') = classification_window_correct;
    history_episode('classification_window_incorrect') = classification_window_incorrect;
    
    is_correct_class = mode(predictions) == real_class;
    history_episode('classification_class_correct') = is_correct_class;
    history_episode('classification_class_incorrect') = ~is_correct_class;
    
    
    history_episode('classification_recognition_correct') = 1;
    history_episode('classification_recognition_incorrect') = 0;
    
end