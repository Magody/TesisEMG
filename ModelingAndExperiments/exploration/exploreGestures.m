load('user55/userData.mat');
training = userData.training;

%{
userData.training{1, 1}.emg, 
userData.training{1, 1}.pointGestureBegins, 
userData.training{1, 1}.pose_myo, 
userData.training{1, 1}.gyro, 
userData.training{1, 1}.accel, 
userData.training{1, 1}.gestureName

%}

lengthTraining = length(training);

listGestures = strings(0,0);

for i=1:lengthTraining
    categoricalGestureName = training{i,1}.gestureName;
    stringGestureName = string(categoricalGestureName);
    
    if sum(listGestures == stringGestureName) == 0
        listGestures = [listGestures, stringGestureName];
    end
end

%% plot channels
emg = training{70, 1}.emg;

%{
hold on;
for channel=1:8
   plot(emg(:, channel)); 
end
hold off;
%}
plot(emg);
xlim([0 length(emg)])
ylim([-1.2 1.2])
grid on

% rectangle
point_start = 400;
window_size = 200;
rect = findall(gcf,'Type', 'rectangle');
delete(rect);
rectangle('Position',[point_start -1  window_size 2],'EdgeColor','r')
        
%% min and max

minimum = 0;
maximum = 0;
for gesture=1:150
    emg = training{gesture, 1}.emg;
    minimum_gesture = min(min(emg));
    maximum_gesture = max(max(emg));
    if minimum_gesture < minimum
       minimum = minimum_gesture;
    end
    
    if maximum_gesture > maximum
       maximum = maximum_gesture;
    end
end
fprintf("Max: %.4f, Min: %.4f", maximum, minimum);

%% signal to matrix map
emg = training{26, 1}.emg;
signal_length = size(emg, 1);

range = 200;

signal_matrix = ones([2*range, 1000]);
signal_round = round(emg*(range-1)) + (range+1);

for channel=1:8
    for i=1:signal_length
        % signal_round(i, channel)
        signal_matrix(signal_round(i, channel), i) = 0;
    end
end

figure(1);
imshow(signal_matrix);
figure(2);
plot(emg);


