clc;

path_root = "/home/magody/programming/MATLAB/tesis/";

%% Parameters and data
history_online1 = load(path_root + "App/matlab/data/saves/save_training_known-nopinchnoopen.mat");
%history_online2 = load(path_root + "App/matlab/data/saves/save_training-nopinchnoopen-2.mat");
history_experiments = containers.Map();
history_experiments1 = history_online1.history_experiments;
%history_experiments2 = history_online2.history_experiments;

keys1 = keys(history_experiments1);
%keys2 = keys(history_experiments2);

for k=1:length(keys1)
    key = keys1{k};
    history_experiments(key) = history_experiments1(key);
end
%{
for k=1:length(keys2)
    key = keys2{k};
    history_experiments(key) = history_experiments2(key);
end
%}
users_len = length(history_experiments);

%% Stats

iterations = 20;

total_classification = zeros([1, iterations]);
total_recognition = zeros([1, iterations]);

max_classification = zeros([1, users_len]);
max_recognition = zeros([1, users_len]);

min_classification = zeros([1, users_len]);
min_recognition = zeros([1, users_len]);

keys_users = keys(history_experiments);

for i=1:users_len
    key = keys_users{i};
    summary = history_experiments(key);
    
    for j=1:length(summary.confusion_matrix)
        add_gesture = summary.add_gesture{j};
        add_reward = summary.add_reward(j);
        fprintf("%s %d\n\n", add_gesture, add_reward);
        confusion_matrix = summary.confusion_matrix{j};
        plotConfMat(confusion_matrix, summary.labels);
        disp("");
    end
    
 
    data_classification = summary.classification(1:iterations);
    data_recognition = summary.recognition(1:iterations);

    total_classification = total_classification + data_classification;
    total_recognition = total_recognition + data_recognition;
 
    max_classification(1, i) = max(data_classification);
    max_recognition(1, i) = max(data_recognition);
 
    min_classification(1, i) = min(data_classification);
    min_recognition(1, i) = min(data_recognition);
 
 
 
end

mean_classification = total_classification ./ users_len;
mean_recognition = total_recognition ./ users_len;

%% Plot
color_classification = [0, 1, 0];
color_recognition = [0, 0, 1];

color_max = [1, 1, 0];
color_min = [1, 0, 0];

figure(1);
subplot(1,2,1);
plot(mean_classification, 'Color', color_classification);
title('Mean classification'); 
xlabel('Interaction');
ylabel('Accuracy');
subplot(1,2,2);
plot(mean_recognition, 'Color', color_recognition);
title('Mean recognition'); 
xlabel('Interaction');
ylabel('Accuracy');

figure(2);
subplot(1,2,1);
hold on;
plot(max_classification, 'Color', color_max);
plot(min_classification, 'Color', color_min);
hold off;
title('Max and min classification'); 
xlabel('Interaction');
ylabel('Accuracy');
subplot(1,2,2);
hold on;
plot(max_recognition, 'Color', color_max);
plot(min_recognition, 'Color', color_min);
hold off;
title('Max and min recognition'); 
xlabel('Interaction');
ylabel('Accuracy');

figure(3);
users_sample = ["user1", "user102", "user223", "user303"];
for j=1:length(users_sample)
 user = users_sample(j);
 summary = history_experiments(user);
 subplot(2,2,j);
 hold on;
 plot(summary.classification, 'Color', color_classification);
 plot(summary.recognition, 'Color', color_recognition);
 hold off;
 t = sprintf("%s classification and recognition", user);
 title(t); 
 xlabel('Interaction');
 ylabel('Accuracy');
end




