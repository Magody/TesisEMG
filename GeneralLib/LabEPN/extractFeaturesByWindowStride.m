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
assignin('base', 'low_umbral', low_umbral);
assignin('base', 'high_umbral', high_umbral);

% we use OfflineDataExp4S, the other experiments are better? (actual
% experiment = 4S
sequence_=WM_X(order);

emgRot        = emg(:,WM_X(user_orientation{5})); % ROTATION
emgCorrect        = emgRot(:,sequence_);  % CORRECTION   

for window=1:num_windows
    startWave = (window-1) * stride;
    endWave = startWave + window_size;          
    features = getFeatures(emgCorrect((startWave+1):endWave, :)); % +1 for matlab
    features_per_window(window, :) = table2array(features);
end


