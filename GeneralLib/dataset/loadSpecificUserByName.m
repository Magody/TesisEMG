function userData = loadSpecificUserByName(folder_name, data_dir)
%LOADSPECIFICUSER Summary of this function goes here
%   Detailed explanation goes here


path_user_data=(horzcat(data_dir,'Specific/',char(folder_name),'/','userData.mat'));
myload = load(path_user_data);
userData = myload.userData;

end

