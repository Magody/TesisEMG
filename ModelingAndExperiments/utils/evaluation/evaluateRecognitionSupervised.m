function history_episodes = evaluateRecognitionSupervised(neural_network, type_execution, dataset_testing, context, verbose_level)
    dataset_testing_length = length(dataset_testing);
    shape_history = [1, dataset_testing_length];
    
    is_validation_or_test = type_execution ~= 1;
    is_train_only = type_execution == 1;
    is_validation_only = type_execution == 2;
    is_test_only = type_execution == 3;

    history_classification_window_correct = zeros(shape_history);
    history_classification_window_incorrect = zeros(shape_history);

    history_classification_correct = zeros(shape_history);
    history_classification_incorrect = zeros(shape_history);

    history_recognition_correct = zeros(shape_history);
    history_recognition_incorrect = zeros(shape_history);
    history_responses = cell(shape_history);

    index_id_user = 1;

    for index_gesture=1:dataset_testing_length
        gesture = dataset_testing{index_gesture};
        
        ground_truth_index = [0, 0];
        if is_test_only
            context('gestureName') = "unknown";
        else
            context('gestureName') = gesture.gestureName;
            

            if isfield(gesture, 'groundTruthIndex')
                ground_truth_index = gesture.groundTruthIndex;
            end
        end

        context('features_per_window') = gesture.("features_per_windowWin"+context('window_size')+"Stride"+context('stride'));


        context('ground_truth_index') = ground_truth_index; 
        context('emg') = gesture.emg;
        history_episode = testRecognitionGestureSupervised(neural_network, type_execution, context, verbose_level-1);

        if is_train_only || is_validation_only
            history_classification_window_correct(index_id_user, index_gesture) = history_episode('classification_window_correct');
            history_classification_window_incorrect(index_id_user, index_gesture) = history_episode('classification_window_incorrect');

            history_classification_correct(index_id_user, index_gesture) = history_episode('classification_class_correct');
            history_classification_incorrect(index_id_user, index_gesture) = history_episode('classification_class_incorrect');

            history_recognition_correct(index_id_user, index_gesture) = history_episode('classification_recognition_correct');
            history_recognition_incorrect(index_id_user, index_gesture) = history_episode('classification_recognition_incorrect');
        end
        history_responses{index_id_user, index_gesture} = history_episode('response');

    end
    history_episodes.history_responses = history_responses;

    history_episodes.history_classification_window_correct = history_classification_window_correct;
    history_episodes.history_classification_window_incorrect = history_classification_window_incorrect;

    history_episodes.history_classification_correct = history_classification_correct;
    history_episodes.history_classification_incorrect = history_classification_incorrect;

    history_episodes.history_recognition_correct = history_recognition_correct;
    history_episodes.history_recognition_incorrect = history_recognition_incorrect;
end

