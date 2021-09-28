clc;
clear all;

seed_rng = 44;
%% Libs

addpath(genpath('../LabEPN'));
addpath(genpath('../utils'));

%% Parameters
fprintf("Parameters config\n");
verbose_level = 0;
RepTraining = 150;
list_users = 1:306; % [1 8];
window_size = 300;
stride = 30;
rangeDown = 1;
dir_data = '/home/magody/programming/MATLAB/tesis/Data/';
prepare_environment(dir_data, verbose_level-1);
assignin('base','RepTraining',  RepTraining); % initial value

%% Generating orientation
generate_rng(seed_rng); 
fprintf("Generating orientation\n");
assignin('base','packetEMG',     false);
Code_0(rangeDown, dir_data);
orientation      = evalin('base', 'orientation');
dataPacket = evalin('base','dataPacket');
num_users = length(list_users);


%% Generating features
fprintf("Generating table of features\n");
generate_rng(seed_rng); 
t_begin = tic;
        
for index_id_user=1:num_users
    user_folder = "user"+list_users(index_id_user);
    userData = loadSpecificUserByName(user_folder, dir_data);
    index_in_packet = getUserIndexInPacket(dataPacket, user_folder);
    assignin('base', 'userIndex', index_in_packet);
    assignin('base','index_user', index_in_packet-2);
    assignin('base','rangeDown', rangeDown);
    assignin('base','emgRepetition', rangeDown);
    
    energy_index = strcmp(orientation(:,1), userData.userInfo.name);
    rand_data=orientation{energy_index,6};
    
    
    features_table = table();
    gestures = {};
    for gesture_number=1:RepTraining
        emgRepetition = evalin('base','emgRepetition');
        if emgRepetition > 150
            emg = userData.testing{rand_data(emgRepetition-150),1}.emg;
            fprintf("Warning: user %d bypass 150", list_users(index_id_user));
        else
            emg = userData.training{rand_data(emgRepetition),1}.emg;
        end
        

        emg_points = length(emg);
        assignin('base','WindowsSize',  window_size);
        assignin('base','Stride',  stride);

        num_windows = getNumberWindows(emg_points, window_size, stride, false);

        for window=1:num_windows

            [~,~,Features_GT,~,~, ~, gestureName, ~, ~] = ...
                Code_1(orientation, dataPacket, RepTraining, verbose_level-1);
            % Features_GT.Properties.RowNames = {char(gestureName)};
            features_table = [features_table; Features_GT];
        end
        gestures{gesture_number, 1} = char(gestureName);
    
    end
    
    model_name = dir_data + "preprocessing" + "/" + "userData" + list_users(index_id_user) +  "Features" + "Win" + window_size + "Stride" + stride + ".mat";
    save(model_name, "features_table", "gestures", "num_windows", "window_size", "stride");
            
end
t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

% features_matrix = table2array(features_table);

