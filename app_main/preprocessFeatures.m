function preprocessFeatures(user_begin, user_end, window_size, stride)

    seed_rng = 44;
    %% Libs

    addpath(genpath('LabEPN'));
    addpath(genpath('utils'));

    %% Parameters
    fprintf("Parameters config\n");
    verbose_level = 0;
    RepTraining = 150;
    RepTesting = 150;
    list_users = user_begin:user_end; % [1 8]; 1:306
    num_users = length(list_users);
    rangeDown = 1;
    dir_data = 'C:\Users\Magody\Documents\GitHub\TesisEMG\Data\'; % '/home/magody/programming/MATLAB/tesis/Data/';
    dit_data_out = dir_data + "preprocessing" + "/";
    assignin('base','RepTraining',  RepTraining); % initial value

    %% Generating orientation
    randn('seed', seed_rng);
    rand('seed', seed_rng);
    fprintf("Generating orientation\n");
    on = true;
    off = false;
    environment_options = struct();
    environment_options.post_processing = on;
    environment_options.randomGestures = off;
    environment_options.noGestureDetection = on;
    environment_options.rangeValues = 300;
    environment_options.packetEMG = false;
    prepare_environment(dir_data, verbose_level-1, environment_options);
    Code_0(rangeDown, dir_data);
    orientation      = evalin('base', 'orientation');
    dataPacket = evalin('base','dataPacket');
    model_orientation = dit_data_out + "orientation" + "Win" + window_size + "Stride" + stride + ".mat";
    save(model_orientation, "orientation");


    %% Generating features
    fprintf("Generating table of features\n");
    t_begin = tic;

    dataPacketSize = length(dataPacket);

    for index_id_user=1:num_users
        user_folder = "user"+list_users(index_id_user);
        fprintf("User %s\n", user_folder);

        user_full_dir = dit_data_out + user_folder + "/";
        % [~, ~, ~] = mkdir(user_full_dir);


        userData = loadUserByNameAndDir(user_folder, char(dit_data_out));
        index_in_packet = getUserIndexInPacket(dataPacket, user_folder);
        assignin('base', 'userIndex', index_in_packet);
        assignin('base','index_user', index_in_packet-2);
        assignin('base','rangeDown', rangeDown);
        assignin('base','emgRepetition', rangeDown);

        energy_index = strcmp(orientation(:,1), userData.userInfo.name);
        rand_data=orientation{energy_index,6};


        training = cell([1, RepTraining]);

        for gesture_number=1:RepTraining


            % emgRepetition = evalin('base','emgRepetition');

            user_gesture = userData.training{rand_data(gesture_number),1};
            emg = user_gesture.emg;    
            emg_points = length(emg);
            assignin('base','WindowsSize',  window_size);
            assignin('base','Stride',  stride);

            num_windows = getNumberWindows(emg_points, window_size, stride, false);

            gesture_struct = user_gesture;
            % gesture_struct.num_windows = num_windows;



            features_per_window = zeros([num_windows, 40]);

            for window=1:num_windows

                [~,~,Features_GT,~,~, ~, ~, ~, ~] = ...
                    Code_1(orientation, dataPacketSize, RepTraining, verbose_level-1);
                
                features_per_window(window, :) = table2array(Features_GT);
            end

            gesture_struct.("features_per_window" + "Win" + window_size + "Stride" + stride) = features_per_window;

            training{gesture_number} = gesture_struct;

        end

        testing = cell([1, RepTesting]);
        % emgRepetition = gesture number?

        for gesture_number=RepTraining+1:RepTraining+RepTesting

            user_gesture = userData.testing{rand_data(gesture_number)-RepTraining,1};
            if gesture_number == RepTraining+RepTesting
                disp("");
            end
            emg = user_gesture.emg;    
            emg_points = length(emg);
            assignin('base','WindowsSize',  window_size);
            assignin('base','Stride',  stride);

            num_windows = getNumberWindows(emg_points, window_size, stride, false);

            gesture_struct = user_gesture;
            % gesture_struct.num_windows = num_windows;



            features_per_window = zeros([num_windows, 40]);

            for window=1:num_windows

                [~,~,Features_GT,~,~, ~, gestureName, ~, ~] = ...
                    Code_1(orientation, dataPacketSize, RepTraining, verbose_level-1);
                % Features_GT.Properties.RowNames = {char(gestureName)};
                features_per_window(window, :) = table2array(Features_GT);
            end

            gesture_struct.("features_per_window" + "Win" + window_size + "Stride" + stride) = features_per_window;

            testing{gesture_number-RepTraining} = gesture_struct;

        end

        userInfo = userData.userInfo;
        sync = userData.sync;

        model_name = user_full_dir + "userData" + ".mat";
        save(model_name, "userInfo", "sync", "training", "testing");

    end




    t_end = toc(t_begin);
    fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

end
