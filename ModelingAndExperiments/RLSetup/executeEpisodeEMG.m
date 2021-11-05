function history_episode = executeEpisodeEMG(q_neural_network, t, type_execution, functionGetReward, context, verbose_level) 
    
    
    is_validation_or_test = type_execution ~= 1;
    is_train_only = type_execution == 1;
    is_validation_only = type_execution == 2;
    
    
    if is_train_only
        q_neural_network.episode = q_neural_network.episode + 1;
    end
    
    
    
    classes_num_to_name = context('classes_num_to_name');
    window_size = context('window_size');
    stride = context('stride');
    
    
    if is_train_only || is_validation_only
        gestureName = context('gestureName');
        ground_truth_index = context('ground_truth_index');
    else
        gestureName = "unknown";
    end
    
    
    noGestureDetection  = context('noGestureDetection');
    is_preprocessed = context('is_preprocessed');
    
    if is_preprocessed
        
        features_per_window = context('features_per_window');
        
        %{
        if gestureName ~= "noGesture"
            
            data_mean = mean(features_per_window, 1);
            data_std = std(features_per_window, 1, 1);
            features_per_window = standardizationCustom(features_per_window, "vertical2D", data_mean, data_std);
        
            data_mean = 0.1; % mean(features_per_window, 1);
            data_std = 0.1; % std(features_per_window, 1, 1);
            features_per_window = standardizationCustom(features_per_window, "vertical2D", data_mean, data_std);
        
        end
        %}
        
        
        num_windows = context('num_windows');
        emg_points = context('emg_points');
    else
        emg = context('emg');
        emg_points = size(emg, 1);
        emg_channels = size(emg, 2);
        emg = reshape(emg, [1, emg_points, emg_channels]);
        num_windows = getNumberWindows(emg_points, window_size, stride, false);
    end
    
    if is_train_only || is_validation_only
        
        ground_truth_gt = getGroundTruthGT(emg_points, ground_truth_index);
        
        
        gt_gestures_pts=zeros([1,num_windows]);
        gt_gestures_pts(1,1)=window_size;

        for k = 1:num_windows-1
            gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
        end
        
        part_of_ground_truth_to_identify = context("part_of_ground_truth_to_identify");

        gt_gestures_labels = mapGroundTruthToLabelsWithPts(gt_gestures_pts, ground_truth_index, gestureName, part_of_ground_truth_to_identify);
        gt_gestures_labels_num = mapGestureLabelsToNumbers(num_windows, gt_gestures_labels, context('classes_name_to_num'));
        
    end
    
    history_episode = containers.Map();
    update_counter = 0;
    update_costs = [];
    reward_cummulated = 0;
    
    classification_window_correct = 0;
    classification_window_incorrect = 0;
    
    
    step_t = 0;
    
    
    expected_num_windows = getNumberWindows(999, window_size, stride, false);
    
    
    etiquetas_labels_predichas_vector = strings([expected_num_windows-1, 1]);
    ProcessingTimes_vector=zeros([1, expected_num_windows-1]);
    TimePoints_vector=zeros([1, expected_num_windows-1]);
    
    
    tic;
    index_state_begin = 1;
    index_state_end = window_size;
    
    
    for window=1:num_windows-1
        
        step_t = step_t + 1;

        if is_preprocessed
            state = features_per_window(window, :)';
        else
            state = emg(:, index_state_begin:index_state_end, :);
        end
        
        [Qval, action] = q_neural_network.selectAction(state, is_validation_or_test); %#ok<ASGLU>
        
        % store all predictions
        etiquetas_labels_predichas_vector(window, 1) = classes_num_to_name(action);
        
        if is_train_only || is_validation_only
            context('real_action') = gt_gestures_labels_num(window+1);
            % context is modified by reference
            [reward, new_state, ~] = functionGetReward(state, action, q_neural_network.qLearningConfig.rewards, context);
            % in this case, new_state is useless
            finish = window == num_windows-1;

            if reward > 0
                classification_window_correct = classification_window_correct + 1;
            else
                classification_window_incorrect = classification_window_incorrect + 1;
            end
        else
            % get reward from real environment (human)
            context('real_action') = "unknown";
            reward = 0; % unknown reward
        end
        
        
        
        if is_train_only
            q_neural_network.saveExperienceReplay(state, action, reward, new_state, finish);
            
            if mod(step_t, q_neural_network.qLearningConfig.interval_for_learning) == 0
                % update in each step could be very brute
                history_learning = q_neural_network.learnFromExperienceReplay(t, verbose_level);
                if history_learning('learned')
                    update_counter = update_counter + 1;
                    update_costs(1, update_counter) = history_learning('mean_cost');
                end
            end
            
        end

        reward_cummulated = reward_cummulated + reward;

        
        % state = new_state;
        
        ProcessingTimes_vector(1,window) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
        TimePoints_vector(1,window)=index_state_end;
        tic;
        index_state_begin = index_state_begin + stride;
        index_state_end = index_state_end + stride;
        
    end
    
    % fill to standarize dimension
    for window=1:expected_num_windows-1
        if etiquetas_labels_predichas_vector(window) == ""
           etiquetas_labels_predichas_vector(window) = "noGesture"; 
        end
        if ProcessingTimes_vector(window) == 0
           ProcessingTimes_vector(window) = mean(ProcessingTimes_vector(1:num_windows-1)); 
        end
        if TimePoints_vector(window) == 0
            TimePoints_vector(window) = TimePoints_vector(window-1) + stride; 
        end
    end
    
    % class_result_num = mode(predictions);
    etiquetas_labels_predichas_vector_without_NoGesture = strings();
    index_no_gesture = 1;
    for i=1:length(etiquetas_labels_predichas_vector)
        if etiquetas_labels_predichas_vector(i,1)~="noGesture"
            etiquetas_labels_predichas_vector_without_NoGesture(index_no_gesture,1)=etiquetas_labels_predichas_vector(i,1);
            index_no_gesture = index_no_gesture + 1;
        end
    end
    
    
    if length(etiquetas_labels_predichas_vector_without_NoGesture) == 1 && etiquetas_labels_predichas_vector_without_NoGesture{1} == ""
        % All labels were noGesture
        class_result = categorical("noGesture");
    else
        class_result = mode(categorical(etiquetas_labels_predichas_vector_without_NoGesture));
        
        for index_label=1:length(etiquetas_labels_predichas_vector)
            label = etiquetas_labels_predichas_vector(index_label);
            if label ~= "noGesture" && label ~= class_result
                etiquetas_labels_predichas_vector(index_label) = string(class_result);
            end
        end
        
        for index_label=1:length(etiquetas_labels_predichas_vector)
            % process batchs
            label = etiquetas_labels_predichas_vector(index_label);
            
            if label == class_result
                continue;
            end
            label_left = label;
            label_right = label;
            if index_label-1 > 0
                label_left = etiquetas_labels_predichas_vector(index_label-1);
            end
            if index_label+1 <= length(etiquetas_labels_predichas_vector)
                label_right = etiquetas_labels_predichas_vector(index_label+1);
            end
            
            if label_left == label_right && (label_left ~= label || label_right ~= label)
                etiquetas_labels_predichas_vector(index_label) = label_left;
            end
        end
        
        
    end
        
   
    
    response.vectorOfLabels = categorical(etiquetas_labels_predichas_vector');
    response.vectorOfTimePoints = TimePoints_vector;
    response.vectorOfProcessingTimes = ProcessingTimes_vector;
    response.class =  categorical(class_result);

 
    history_episode('reward_cummulated') = reward_cummulated;
    history_episode('response') = response;
    
    
    recognition = 0;
    if is_train_only || is_validation_only
        % GROUND TRUTH (no depende del modelo)------------
        repInfo.gestureName =  gestureName;
        repInfo.groundTruth = ground_truth_gt;
        try
            r1 = evalRecognition(repInfo, response);
            if isnan(r1.overlappingFactor)
                % disp("overlapping nan");
                disp("");
            end
        catch ME
            disp('Error al aplicar reconocimiento');

            r1.recogResult=0;

        end

        
        
        if ~isempty(r1.recogResult)

            if  r1.recogResult==1
                recognition = 1;
            end

        elseif gestureName == "noGesture"
            recognition = r1.classResult;
        end

        if verbose_level > 0
           fprintf("%s: %s, %d\n", string(gestureName), string(class_result), recognition); 
        end
        
        history_episode('update_costs') = update_costs;
    
        history_episode('classification_window_correct') = classification_window_correct;
        history_episode('classification_window_incorrect') = classification_window_incorrect;

        %{
        if (mode(predictions) == real_class) ~= (string(class_result) == string(gestureName))
            disp("Incongruency");
        end
        %}
        is_correct_class = string(class_result) == string(gestureName);
        history_episode('classification_class_correct') = is_correct_class;
        history_episode('classification_class_incorrect') = ~is_correct_class;


        is_correct_recognition = recognition == 1;
        history_episode('classification_recognition_correct') = is_correct_recognition;
        history_episode('classification_recognition_incorrect') = ~is_correct_recognition;
        
    end
    
    
    
end