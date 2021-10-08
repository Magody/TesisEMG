%{

CODES.CONNECT_ROOM = 1003;
CODES.TEST = 1000;

% 2000> -> reserved for MATLAB APP
CODES.CHART = 2001;
CODES.PREDICTION = 2002;

%}







try
    
    client = Client('ws://127.0.0.1:3000');
    
    client.sendCustomMessage(CODES.CONNECT_ROOM, '', strcat('int->', "10000"));
    
    
    
    %%%%% MOCK signal from sensor
    load('/home/magody/programming/MATLAB/tesis/QNN_EMG_modelo_full_RL_Refactoring/Data/Specific/user1/userData.mat');
    emg = userData.training{27, 1}.emg;
    sync = userData.sync;
    pose_myo = userData.training{27, 1}.pose_myo;
    gyro = userData.training{27, 1}.gyro;
    accel = userData.training{27, 1}.accel;
    
    

    % orientation = Code_0_unit(emg, sync, "usuario1");

    client.emg = emg;
    client.sync = sync;
    
    
    pause(20);
    
    client.close();
    
    %{
    
    
    %}
    disp("END");
    
catch error
    disp(error);
    client.close();
end


