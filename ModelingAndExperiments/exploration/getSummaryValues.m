load('user262/userData.mat');
training = userData.training;
testing = userData.testing;

%{
userData.training{1, 1}.emg, 
userData.training{1, 1}.pointGestureBegins, 
userData.training{1, 1}.pose_myo, 
userData.training{1, 1}.gyro, 
userData.training{1, 1}.accel, 
userData.training{1, 1}.gestureName

%}

mapGesture = containers.Map;



lengthTraining = length(training);
listGesturesTraining = strings(0,0);
listGesturesTesting = strings(0,0);

for i=1:lengthTraining
    trainingCategoricalGestureName = training{i,1}.gestureName;
    trainingStringGestureName = string(trainingCategoricalGestureName);
    
    testingCategoricalGestureName = testing{i,1}.gestureName;
    testingStringGestureName = string(testingCategoricalGestureName);
    
    key_training = strcat("training_",trainingStringGestureName);
    key_testing = strcat("testing_",testingStringGestureName);
    
    if sum(listGesturesTraining == trainingStringGestureName) == 0
        listGesturesTraining = [listGesturesTraining, trainingStringGestureName];
        
        mapGesture(key_training) = [];
        
    end
    
    if sum(listGesturesTesting == testingStringGestureName) == 0
        listGesturesTesting = [listGesturesTesting, testingStringGestureName];
        mapGesture(key_testing) = [];
        
    end        
    
    mapGesture(key_training) = ...
    [mapGesture(key_training), ...
     length(training{i,1}.emg)];
    
    mapGesture(key_testing) = ...
    [mapGesture(key_testing), ...
     length(testing{i,1}.emg)];
    
    
    
    
    
end

mapKeys = keys(mapGesture);

for i=1:length(mapKeys)
    key = mapKeys{i};
    value = mapGesture(key);
    value_min = min(value);
    value_max = max(value);
    value_mean = mean(value);
    value_quantity = length(value);
    fprintf("%s has emg[min=%f,max=%f,avg=%f] from %d\n", ...
        key, value_min, value_max, value_mean, value_quantity);
end

