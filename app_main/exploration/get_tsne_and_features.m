clc;
clear all;
addpath(genpath('QNN Toolbox'));

%% Parameters
fprintf("Parameters config\n");
rng('default')
plot_point_size = 5;
verbose_level = 1;
RepTraining = 150;
list_users = 1:306; % [1 8];
rangeDown = 1;
prepare_environment(-1, verbose_level-1);
assignin('base','RepTraining',  RepTraining); % initial value

%% Generating orientation
fprintf("Generating orientation\n");
Code_0(rangeDown);
orientation      = evalin('base', 'orientation');
dataPacket = evalin('base','dataPacket');
num_users = length(list_users);



%% Generating table of features
fprintf("Generating table of features\n");
features_table = table();
gestures = {};
for index_id_user=1:num_users
    user_folder = "user"+list_users(index_id_user);
    userData = loadSpecificUserByName(user_folder);
    index_in_packet = getUserIndexInPacket(dataPacket, user_folder);
    assignin('base', 'userIndex', index_in_packet);
    assignin('base','index_user', index_in_packet-2);
    assignin('base','rangeDown', rangeDown);
    assignin('base','emgRepetition', rangeDown);
    
    energy_index = strcmp(orientation(:,1), userData.userInfo.name);
    rand_data=orientation{energy_index,6};
    
    for gesture_number=1:RepTraining
        emgRepetition = evalin('base','emgRepetition');
        if gesture_number > 150
            emg = userData.testing{rand_data(emgRepetition-150),1}.emg;
        else
            emg = userData.training{rand_data(emgRepetition),1}.emg;
        end
        
        numberPointsEmgData = length(emg);
        assignin('base','WindowsSize',  numberPointsEmgData);
        assignin('base','Stride',  numberPointsEmgData);
                
                
                
        [~,~,Features_GT,~,~, ~, gestureName, ~, ~] = ...
                Code_1(orientation, RepTraining, verbose_level-1);
        % Features_GT.Properties.RowNames = {char(gestureName)};
        index_gesture = ((index_id_user-1) * RepTraining) + gesture_number;
        gestures{index_gesture, 1} = char(gestureName);
        features_table = [features_table; Features_GT];
        disp("");
    end
    disp("");
            
end

features_matrix = table2array(features_table);
%% Preprocessing for plot
fprintf("Preprocessing for plot\n");

gestures_unique = unique(gestures);
random_colors = {[1 0 0], [0 1 0], [0 0 1], [1 1 0], [1 0 1], [0 1 1]};
%{
for i=1:length(gestures_unique)
    random_colors{i} = rand([1 3]);
end
%}
gestures_colors = containers.Map(gestures_unique, random_colors);

colors = [];
for i=1:length(gestures)
    gesture_name = gestures{i};
    colors = [colors; gestures_colors(gesture_name)];
end




%% TSNE

algorithms = [ ...
    struct('distance', 'hamming','plot', struct('title', 'Hamming')), ...
    struct('distance', 'mahalanobis','plot', struct('title', 'Mahalanobis')), ...
    struct('distance', 'chebychev','plot', struct('title', 'Chebychev')), ...
    struct('distance', 'euclidean','plot', struct('title', 'Euclidean')), ...
    ];

len_algorithms = length(algorithms);

fprintf("TSNE for %d algorithms\n", len_algorithms);

% plot_rows = ceil((len_algorithms)/2);
% plot_cols = floor((len_algorithms+1)/2);

fig_count = 1;

reduced_features_matrix = features_matrix(1:10000, :);
reduced_gestures = gestures(1:10000, :);
reduced_colors = colors(1:10000, :);

for i=1:len_algorithms
    figure(fig_count);
    [Y_2D, loss_2D] = tsne(reduced_features_matrix,'Algorithm','exact', ...
             'Distance',algorithms(i).distance, 'NumDimensions',2);
    
    % subplot(plot_rows, plot_cols,i)
    gscatter(Y_2D(:,1),Y_2D(:,2),reduced_gestures, reduced_colors, '.', plot_point_size)
    title(algorithms(i).plot.title)
    
    
    saveas(gcf, ['figures/' algorithms(i).plot.title '_2D.png']);
    
    figure(fig_count+1);
    [Y_3D, loss_3D] = tsne(reduced_features_matrix,'Algorithm','exact', ...
             'Distance',algorithms(i).distance, 'NumDimensions',3);
         
    v = double(categorical(gestures));
    % subplot(plot_rows, plot_cols,i)
    scatter3(Y_3D(:,1),Y_3D(:,2),Y_3D(:,3),plot_point_size, reduced_colors,'filled')
    title(algorithms(i).plot.title)
    
    fig_count = fig_count + 2;
    
	saveas(gcf, ['figures/' algorithms(i).plot.title '_3D.png']);
    
    fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n', loss_2D, loss_3D)

    
end



%{
Y = tsne(features_matrix,'Algorithm','exact','Distance','cosine');
subplot(2,2,2)
gscatter(Y(:,1),Y(:,2),gestures)
title('Cosine')
%}

