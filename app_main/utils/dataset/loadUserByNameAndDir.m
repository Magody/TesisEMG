function userData = loadUserByNameAndDir(folder_name, data_dir, is_legacy)
%LOADSPECIFICUSER Summary of this function goes here
%   Detailed explanation goes here

    if nargin == 2
       is_legacy = true; 
    end


    path_user_data=(horzcat(data_dir,char(folder_name),'/','userData.mat'));
    myload = load(path_user_data);
    if is_legacy
        userData = myload.userData;
    else
        userData = myload;
    end
end

