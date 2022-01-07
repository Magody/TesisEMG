%% libs
path_root = "C:/Git/TesisEMG/";
path_to_framework = "C:/Git/MATLABMagodyFramework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";

path_to_data_for_train = horzcat(char(path_root),'Data/preprocessing/');

addpath(genpath(path_root + "GeneralLib"));
addpath(genpath(path_root + "ModelingAndExperiments"));

%%
win = 1000;
user_folder_name = "user309";

user0_original = load('C:\Git\TesisEMG\Data\preprocessing\user01/LorenaBaronaDate2\userData.mat');
user0 = load('C:\Git\TesisEMG\Data\preprocessing\user37\userData.mat');
user0.userInfo.name = user_folder_name;
user0.userInfo.username = user_folder_name;
user0.userInfo.age = user0_original.userData.userInfo.age;
user0.userInfo.gender = "woman";
user0.userInfo.occupation = "teacher";
user0.userInfo.ethnicGroup = "latin";
user0.userInfo.handedness = "right";
user0.userInfo.ArmDamage = "False";
user0.userInfo.distanceFromElbowToMyoInCm = 5;
user0.userInfo.distanceFromElbowToUlnaInCm = 23;
user0.userInfo.armPerimeterInCm = 22.5;
user0.userInfo.date = "30-Oct-2021 14:34:28";

sync = user0_original.userData.gestures.sync.data;

%%
gestures = user0_original.userData.gestures;

data_train = struct();
data_train_names = containers.Map(1:25:150,["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]);
data_train.id1 = gestures.relax.data(1:25);
data_train.id26 = gestures.fist.data(1:25);
data_train.id51 = gestures.waveIn.data(1:25);
data_train.id76 = gestures.waveOut.data(1:25);
data_train.id101 = gestures.open.data(1:25);
data_train.id126 = gestures.pinch.data(1:25);

resp1 = user0.training;

for from=1:25:150
    disp(from);
    dataset = customGenerate(data_train.("id"+from), resp1, from, string(data_train_names(from)), win);
    
    resp1(1, from:(from+24)) = dataset(:);
end


%%

data_train = struct();
data_train_names = containers.Map(1:25:150,["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]);
data_train.id1 = gestures.relax.data(26:30);
data_train.id26 = gestures.fist.data(26:30);
data_train.id51 = gestures.waveIn.data(26:30);
data_train.id76 = gestures.waveOut.data(26:30);
data_train.id101 = gestures.open.data(26:30);
data_train.id126 = gestures.pinch.data(26:30);

resp2 = user0.training;

for from=1:25:150
    disp(from);
    dataset = customGenerate(data_train.("id"+from), resp2, from, string(data_train_names(from)),win);
    
    resp2(1, from:(from+24)) = dataset(:);
end

%%

user0.training = resp1;
user0.testing = resp2;


training = user0.training;
testing = user0.testing;
userInfo = user0.userInfo;
% sync = user0.sync;

save(path_root+"Data\preprocessing\" + user_folder_name +"\userData.mat", "userInfo", "sync", "training", "testing");



%%
t1 = tic;
for i=80:85
    features_per_window100 = extractFeaturesByWindowStride(path_root, orientation100, 300, 30, user100.training{i}.emg);
    data_mean = mean(features_per_window100, 1);
    data_std = std(features_per_window100, 1, 1);
end
t2 = toc(t1);

 
fprintf("Total Elapsed time: %.4f [ms]\n", (t2/5)*1000);

%% t
