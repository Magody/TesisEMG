 function giveFeedback(app, reward)
            
    global qnn_online context_online
    global last_sample last_prediction
    global window_size stride
    global last_reward;
            
    last_reward = reward;
                
    
    features_name = "features_per_windowWin" + window_size + "Stride" + stride;
    
    last_sample.features_name = features_name;
    states = last_sample.(features_name);
    num_windows = size(states, 1);
    
    gt_gestures_pts=zeros([1,num_windows]);
    gt_gestures_pts(1,1)=window_size;

    for k = 1:num_windows-1
        gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
    end
    
    classes_name_to_num = context_online('classes_name_to_num');
    gt_gestures_labels = mapGroundTruthToLabelsWithPts(gt_gestures_pts, last_sample.groundTruthIndex, last_prediction, 0.2);
    gt_gestures_labels_num = mapGestureLabelsToNumbers(num_windows, gt_gestures_labels, classes_name_to_num);

    % only is labeled the ground truth part
    
    action_no_gesture = classes_name_to_num("noGesture");
    
    for index_state=1:num_windows
        action = gt_gestures_labels_num(index_state);
        %{
        if action == action_no_gesture
            % we dont know if was correct or incorrect
            continue;
        end
        %}
        state = states(index_state, :);
        new_state = state;
        is_terminal = index_state == num_windows;
        qnn_online.saveExperienceReplay(state', action, last_reward, new_state', is_terminal)
        
    end
    
    global extra_dataset index_extra_dataset class_added
    global last_gesture stored_extra_classes
    
    % classes_name_to_num = context_online('classes_name_to_num');
    
    
    classes = ["waveOut", "waveIn", "fist", "noGesture"];
        
    classes_merged = [classes, stored_extra_classes];
    
    add_to_dataset = true;
    global real_gestureName
    for i=1:length(classes)
        if classes(i) == string(real_gestureName) % string(last_gesture.gestureName)
            add_to_dataset = false;
            break;
        end
    end
    
    train_from_dataset = false;
        
    if add_to_dataset
        limit = 50;
        last_sample.gestureName = real_gestureName; % string(last_gesture.gestureName);
        
        data = generateDataAugmentation(null(1), last_sample);
       
        for i=1:length(data)
            extra_dataset{index_extra_dataset} = data{i};
            index_extra_dataset = mod(index_extra_dataset + 1, limit);
            if index_extra_dataset == 0
                index_extra_dataset = limit;
            end
        end
        
        
        train_from_dataset = true;
        
        
    end
    global counter
    counter = counter + 1;
    if mod(counter, 30) == 0
        % train full each intervals
        train_from_dataset = true;
    end
        
    
    learn(null(1), train_from_dataset);
    
end