%% libs
path_root = "/home/magody/programming/MATLAB/tesis/";
path_to_data_for_train = horzcat(char(path_root),'Data/preprocessing/');

%% Process

user100 = load([path_to_data_for_train 'user100/userData.mat']);
orientation100 = getOrientation(user100, "user100");
t1 = tic;
for i=80:85
    features_per_window100 = extractFeaturesByWindowStride(path_root, orientation100, 300, 30, user100.training{i}.emg);
    data_mean = mean(features_per_window100, 1);
    data_std = std(features_per_window100, 1, 1);
end
t2 = toc(t1);

 
fprintf("Total Elapsed time: %.4f [ms]\n", (t2/5)*1000);

%% t
variable_names = {'WMoos_F1_Ms1','WMoos_F1_Ms2','WMoos_F1_Ms3','WMoos_F1_Ms4','WMoos_F1_Ms5','WMoos_F1_Ms6','WMoos_F1_Ms7','WMoos_F1_Ms8','WMoos_F2_Ms1','WMoos_F2_Ms2','WMoos_F2_Ms3','WMoos_F2_Ms4','WMoos_F2_Ms5','WMoos_F2_Ms6','WMoos_F2_Ms7','WMoos_F2_Ms8','WMoos_F3_Ms1','WMoos_F3_Ms2','WMoos_F3_Ms3','WMoos_F3_Ms4','WMoos_F3_Ms5','WMoos_F3_Ms6','WMoos_F3_Ms7','WMoos_F3_Ms8','WMoos_F4_Ms1','WMoos_F4_Ms2','WMoos_F4_Ms3','WMoos_F4_Ms4','WMoos_F4_Ms5','WMoos_F4_Ms6','WMoos_F4_Ms7','WMoos_F4_Ms8','WMoos_F5_Ms1','WMoos_F5_Ms2','WMoos_F5_Ms3','WMoos_F5_Ms4','WMoos_F5_Ms5','WMoos_F5_Ms6','WMoos_F5_Ms7','WMoos_F5_Ms8'};
c = cell([1,40]);
for i=1:40
   c{i} = 0; 
end
t = cell2table(c);
t.Properties.VariableNames = variable_names;
disp(t);