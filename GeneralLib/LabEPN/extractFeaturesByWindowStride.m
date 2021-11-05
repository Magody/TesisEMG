function features_per_window = extractFeaturesByWindowStride(path_root, ...
                                    user_orientation, window_size, stride, emg)
%% Load libs
addpath(genpath(path_root + "GeneralLib"));

%% Get Features
num_windows = getNumberWindows(size(emg, 1), window_size, stride, false);
features_per_window = zeros([num_windows, 40]);

order= user_orientation{2};
low_umbral = user_orientation{3};
high_umbral = user_orientation{4};

% we use OfflineDataExp4S, the other experiments are better? (actual
% experiment = 4S
sequence_=WM_X(order);

emgRot        = emg(:,WM_X(user_orientation{5})); % ROTATION
emgCorrect        = emgRot(:,sequence_);  % CORRECTION   


variable_names = {'WMoos_F1_Ms1','WMoos_F1_Ms2','WMoos_F1_Ms3','WMoos_F1_Ms4','WMoos_F1_Ms5','WMoos_F1_Ms6','WMoos_F1_Ms7','WMoos_F1_Ms8','WMoos_F2_Ms1','WMoos_F2_Ms2','WMoos_F2_Ms3','WMoos_F2_Ms4','WMoos_F2_Ms5','WMoos_F2_Ms6','WMoos_F2_Ms7','WMoos_F2_Ms8','WMoos_F3_Ms1','WMoos_F3_Ms2','WMoos_F3_Ms3','WMoos_F3_Ms4','WMoos_F3_Ms5','WMoos_F3_Ms6','WMoos_F3_Ms7','WMoos_F3_Ms8','WMoos_F4_Ms1','WMoos_F4_Ms2','WMoos_F4_Ms3','WMoos_F4_Ms4','WMoos_F4_Ms5','WMoos_F4_Ms6','WMoos_F4_Ms7','WMoos_F4_Ms8','WMoos_F5_Ms1','WMoos_F5_Ms2','WMoos_F5_Ms3','WMoos_F5_Ms4','WMoos_F5_Ms5','WMoos_F5_Ms6','WMoos_F5_Ms7','WMoos_F5_Ms8'};
c = cell([1,40]);
for i=1:40
   c{i} = 0; 
end
empty_table = cell2table(c);
empty_table.Properties.VariableNames = variable_names;


for window=1:num_windows
    startWave = (window-1) * stride;
    endWave = startWave + window_size;          
    features = getFeatures(emgCorrect((startWave+1):endWave, :), empty_table, low_umbral, high_umbral); % +1 for matlab
    features_per_window(window, :) = table2array(features);
end



end


