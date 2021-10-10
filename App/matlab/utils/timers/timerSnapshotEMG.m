function timerSnapshotEMG(~, ~) % timerObject, timerInfo
    global deviceType path_root myoObject window_size stride orientation sample_time_ms model context;
    
    emg_stored_length = size(myoObject.myoData.emg_log, 1);
    
    if emg_stored_length < sample_time_ms
        % not enought data
        return; 
    end
    sample_begin = emg_stored_length-sample_time_ms;
    emg = myoObject.myoData.emg_log((sample_begin+1):emg_stored_length, :);
    
    tic_toc_feature_extraction = tic;
    features_per_window = extractFeaturesByWindowStride(path_root, orientation, window_size, stride, emg);
    time_used_for_feature_extraction_ms = toc(tic_toc_feature_extraction) * 1000;
    
    features_name = "features_per_windowWin" + window_size + "Stride" + stride;
    
    sample = struct("emg", emg, features_name, features_per_window);
    context('user_gestures_test') = {sample};
    
    % fprintf("Snapshot prepared, predicting...!\n")
    % disp(mean(emg(:)));
    tic_toc_qnn_prediction = tic;
    history_test = model.q_neural_network.runEpisodes(@getRewardEMG, 3, context, 0);  
    time_used_for_qnn_prediction_ms = toc(tic_toc_qnn_prediction) * 1000 ;
    
    time_used_total_ms = time_used_for_feature_extraction_ms + time_used_for_qnn_prediction_ms;
    
    fprintf("*****\nPrediction %s in %.2f[ms]. \n->%.2f[ms] (%.1f %%) used in feature extraction and \n->%.2f[ms] (%.1f %%) used in qnn prediction\n", ...
        string(history_test.history_responses{1}.class), ...
        time_used_total_ms, ...
        time_used_for_feature_extraction_ms, ...
        100*time_used_for_feature_extraction_ms/time_used_total_ms, ...
        time_used_for_qnn_prediction_ms, ...
        100*time_used_for_qnn_prediction_ms/time_used_total_ms)

end

