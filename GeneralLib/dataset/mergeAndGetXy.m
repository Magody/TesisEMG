function [X, y] = mergeAndGetXy(dataset, window_size, stride, classes_name_to_num)
    % dataset is a 1xn cell
    X = [];
    y = [];
    
    for i=1:length(dataset)
        gesture = dataset{i};
        gestureName = string(gesture.gestureName);
        
        
        features_per_window = gesture.("features_per_windowWin"+window_size+"Stride"+stride);
    
        num_windows = size(features_per_window, 1);
        
        ground_truth_index = [0, 0];
        
        if isfield(gesture, 'groundTruthIndex')
            ground_truth_index = gesture.groundTruthIndex;
        end
        
        
        gt_gestures_pts=zeros([1,num_windows]);
        gt_gestures_pts(1,1)=window_size;
        for k = 1:num_windows-1
            gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
        end
        
        part_of_ground_truth_to_identify = 0.2;

        gt_gestures_labels = mapGroundTruthToLabelsWithPts(gt_gestures_pts, ground_truth_index, gestureName, part_of_ground_truth_to_identify);
        gt_gestures_labels_num = mapGestureLabelsToNumbers(num_windows, gt_gestures_labels, classes_name_to_num);
        
        
        
        one_hot = sparse_one_hot_encoding(gt_gestures_labels_num, classes_name_to_num.Count);
        X = [X, features_per_window'];
        y = [y; one_hot];
        
    end
    

end

