%% libs
path_root = "C:/Git/TesisEMG/";
path_to_framework = "C:/Git/MATLABMagodyFramework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";

path_to_data_for_train = horzcat(char(path_root),'Data/preprocessing/');

addpath(genpath(path_root + "GeneralLib"));

%%
userDanny0 = load('C:\Git\TesisEMG\Data\preprocessing\user37\userData.mat');
userDanny0.name = "user307";
userDanny0.username = "user307";
userDanny0.userInfo.distanceFromElbowToUlnaInCm = 26;
userDanny0.userInfo.armPerimeterInCm = 24;
userDanny0.userInfo.date = "19-Jan-2018 14:20:53";


%%
user0_train = load('C:\Git\TesisEMG\Data\preprocessing\user0\userData.mat');
user0_test = load('C:\Git\TesisEMG\Data\preprocessingTest\user0\userData.mat');

%%
d1 = user0_train.userDataTrain.gestures;

data_train = struct();
data_train_names = containers.Map(1:25:150,["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]);
data_train.id1 = d1.relax.data;
data_train.id26 = d1.fist.data;
data_train.id51 = d1.waveIn.data;
data_train.id76 = d1.waveOut.data;
data_train.id101 = d1.fingersSpread.data;
data_train.id126 = d1.doubleTap.data;

resp1 = userDanny0.training;

for from=1:25:150
    disp(from);
    dataset = customGenerate(data_train.("id"+from), resp1, from, string(data_train_names(from)));
    
    resp1(1, from:(from+24)) = dataset(:);
end


%%
d2 = user0_test.userDataTest.gestures;

data_train = struct();
data_train_names = containers.Map(1:25:150,["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]);
data_train.id1 = d2.relax.data;
data_train.id26 = d2.fist.data;
data_train.id51 = d2.waveIn.data;
data_train.id76 = d2.waveOut.data;
data_train.id101 = d2.fingersSpread.data;
data_train.id126 = d2.doubleTap.data;

resp2 = userDanny0.training;

for from=1:25:150
    disp(from);
    dataset = customGenerate(data_train.("id"+from), resp2, from, string(data_train_names(from)));
    
    resp2(1, from:(from+24)) = dataset(:);
end

%%

userDanny0.training = resp1;
userDanny0.testing = resp2;


training = userDanny0.training;
testing = userDanny0.testing;
userInfo = userDanny0.userInfo;
sync = userDanny0.sync;


save(path_root+"Data\preprocessing\user307\userData.mat", "userInfo", "sync", "training", "testing");



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
