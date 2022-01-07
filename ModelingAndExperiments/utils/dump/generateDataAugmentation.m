function data = generateDataAugmentation(app, sample)
    global path_root orientation window_size stride
    
    features_name = sample.features_name;
    
    data_augmentation_moving_avg_box = {};
    for i=7:15
        emg_new = movmean(sample.emg, i);
        features_per_window = extractFeaturesByWindowStride(path_root, orientation, window_size, stride, emg_new);
        
        data_augmentation_moving_avg_box = [data_augmentation_moving_avg_box, ...
            {struct("emg", emg_new, ...
                features_name, features_per_window, ...
                "groundTruthIndex", sample.groundTruthIndex, ...
                "gestureName", sample.gestureName)}]; %#ok<*AGROW> 
        
    end
    
    data_augmentation_shift = {};
    for i=5:10
        shift = i*5;
        emg_length = size(sample.emg, 1);
        emg_new_left = zeros(size(sample.emg));
        emg_new_right = zeros(size(sample.emg));
        emg_new_left(shift:emg_length, :) = sample.emg(1:(emg_length-shift+1), :);
        groundTruthLeft = [0, 0];
        groundTruthLeft(1) = min(sample.groundTruthIndex(1) + shift, emg_length);
        groundTruthLeft(2) = min(sample.groundTruthIndex(2) + shift, emg_length);
        
        emg_new_right(1:(emg_length-shift+1), :) = sample.emg(shift:emg_length, :);
        groundTruthRight = [0, 0];
        groundTruthRight(1) = max(sample.groundTruthIndex(1) - shift, 1);
        groundTruthRight(2) = max(sample.groundTruthIndex(2) - shift, 1);
        
        
        features_per_window_left = extractFeaturesByWindowStride(path_root, orientation, window_size, stride, emg_new_left);
        
        data_augmentation_shift = [data_augmentation_shift, ...
            {struct("emg", emg_new_left, ...
            features_name, features_per_window_left, ...
            "groundTruthIndex", groundTruthLeft, ...
            "gestureName", sample.gestureName)}];
        
        features_per_window_right = extractFeaturesByWindowStride(path_root, orientation, window_size, stride, emg_new_right);
        
        data_augmentation_shift = [data_augmentation_shift, ...
            {struct("emg", emg_new_right, ...
            features_name, features_per_window_right, ...
            "groundTruthIndex", groundTruthRight, ...
            "gestureName", sample.gestureName)}];
        
    end
    
    data = [{sample}, ...
        data_augmentation_moving_avg_box, ...
        data_augmentation_shift];
end