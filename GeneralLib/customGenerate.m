function dataset = customGenerate(data, resp, from, class)

    
    
    data_len = length(data);
    dataset = {};
    for i=1:25
        index = mod(from+i-1, data_len);
        if index == 0
            index = data_len;
        end
        real_index = from+i-1;
    
        
        % user0.userDataTrain.gestures.fingersSpread.data{1, 1}.rot
    
        dataset{1, i}.emg = data{index}.emg;

        dataset{1, i}.pose_myo = [];
        if isfield(dataset{1, i}, "pose_myo")
            dataset{1, i}.pose_myo = data{index}.pose_myo;
        end
         
        dataset{1, i}.gyro = resp{real_index}.gyro;
        dataset{1, i}.accel = resp{real_index}.accel;
        
        dataset{1, i}.gestureName = categorical(class);
        if class == "noGesture"
            dataset{1, i}.groundTruthIndex = [0, 0]; % data{index}.emgGroundTruthIndex
            dataset{1, i}.groundTruth = zeros([1, 1000]); % data{index}.emgGroundTruth
        else

            if isfield(data{index}, "emgGroundTruthIndex")
                dataset{1, i}.groundTruthIndex = data{index}.emgGroundTruthIndex;
                dataset{1, i}.groundTruth = data{index}.emgGroundTruth;
            else
                index_begin = data{index}.pointGestureBegins;
                emg_length = length(data{index}.emg);
                dataset{1, i}.groundTruthIndex = [index_begin, min(index_begin+250, emg_length)];
                dataset{1, i}.groundTruth = zeros([1, emg_length]);
                dataset{1, i}.groundTruth(dataset{1, i}.groundTruthIndex(1):dataset{1, i}.groundTruthIndex(2)) = 1;
                
            end
            
     
        end
        dataset{1, i}.pointGestureBegins = dataset{1, i}.groundTruthIndex(1);
        
    end
end

