function [classification, recognition] = onlineQNNTest(app, event)
    global qnn_online context_online
    global known_dataset_test
    global user_folder
    
    context_online('user_gestures_validation') = [known_dataset_test];
    
    [summary, ~, history_test] = ExperimentHelper.testModelIndividual(qnn_online, true, context_online, 0);

    global labels;
    % app.showConfusionMatrix(history_test.confusion_matrix, labels);
    classification = summary.classification_test*100;
    recognition = summary.recognition_test*100;
    
    global last_prediction_complete last_reward;
    %{
    history_user = app.history_experiments(user_folder);
    history_user.classification = [history_user.classification, classification];
    history_user.recognition = [history_user.recognition, recognition];
    history_user.labels = labels;
    history_user.add_gesture = [history_user.add_gesture, last_prediction_complete];
    history_user.add_reward = [history_user.add_reward, last_reward];
    history_user.confusion_matrix = [history_user.confusion_matrix, history_test.confusion_matrix];
    
    app.history_experiments(user_folder) = history_user;
    plot(app.UIAxesOnlineEvolutionClassification, history_user.classification);
    plot(app.UIAxesOnlineEvolutionRecognition, history_user.recognition);
    
    
    app.count_test = app.count_test + 1;
    app.LabelOnlineCountTest.Text = sprintf("Count: %d", app.count_test);
    %}

end