function userData = loadSpecificUserWindowStrideByName(full_folder_name, window_size, stride)
%LOADSPECIFICUSER Summary of this function goes here
%   Detailed explanation goes here

path_user_data=(horzcat(char(full_folder_name),'/','userDataFeaturesWin', char(string(window_size)), 'Stride', char(string(stride)) ,'.mat'));
userData = load(path_user_data);

end

