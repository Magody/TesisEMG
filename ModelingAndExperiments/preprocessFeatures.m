function preprocessFeatures(user_begin, user_end, window_size, stride, is_legacy, path_root)

    seed_rng = 44;
    %% Libs

    addpath(genpath(path_root + "GeneralLib/LabEPN"));

    %% Parameters
    fprintf("Parameters config\n");
    RepTraining = 150;
    RepTesting = 150;
    list_users = user_begin:user_end; % [1 8]; 1:306
    num_users = length(list_users);
	if is_legacy
		dir_data = path_root + "Data/"; 
		dit_data_out = dir_data + "preprocessing/";
	else
		dir_data = path_root + "Data/preprocessing/";
		dit_data_out = dir_data;
	end
    
    assignin('base','RepTraining',  RepTraining); % initial value

    %% Generating orientation
    randn('seed', seed_rng); %#ok<*RAND>
    rand('seed', seed_rng);
    on = true;
    off = false;
    environment_options = struct();
    environment_options.post_processing = on;
    environment_options.randomGestures = off;
    environment_options.noGestureDetection = on;
    environment_options.rangeValues = 300;
    environment_options.packetEMG = false;
    prepare_environment(environment_options);

    assignin('base','is_legacy',is_legacy);
    

    %% Generating features
    fprintf("Generating table of features\n");
    t_begin = tic;

    for index_id_user=1:num_users
        user_folder = "user"+list_users(index_id_user);
        fprintf("User %s\n", user_folder);

        if is_legacy
            user_full_dir = dir_data + "Specific/";
            user_full_dir_out = dit_data_out + user_folder + "/";
        else
            user_full_dir = dit_data_out;
            user_full_dir_out = dit_data_out + user_folder + "/";
        end
        
        [~, ~, ~] = mkdir(user_full_dir_out);


        userData = loadUserByNameAndDir(user_folder, char(user_full_dir), is_legacy);
        
        orientation_user = getOrientation(userData, user_folder);

        training = cell([1, RepTraining]);

        for gesture_number=1:RepTraining


            % emgRepetition = evalin('base','emgRepetition');

            user_gesture = userData.training{gesture_number};
            emg = user_gesture.emg;    
            assignin('base','WindowsSize',  window_size);
            assignin('base','Stride',  stride);

            gesture_struct = user_gesture;
            % gesture_struct.num_windows = num_windows;


            features_per_window = extractFeaturesByWindowStride(path_root, orientation_user, window_size, stride, emg);          
            

            gesture_struct.("features_per_window" + "Win" + window_size + "Stride" + stride) = features_per_window;

            training{gesture_number} = gesture_struct;

        end

        testing = cell([1, RepTesting]);
        % emgRepetition = gesture number?

        for gesture_number=RepTraining+1:RepTraining+RepTesting

            user_gesture = userData.testing{gesture_number-RepTraining};
            if gesture_number == RepTraining+RepTesting
                disp("");
            end
            emg = user_gesture.emg;    
            assignin('base','WindowsSize',  window_size);
            assignin('base','Stride',  stride);

            gesture_struct = user_gesture;
            % gesture_struct.num_windows = num_windows;



            features_per_window = extractFeaturesByWindowStride(path_root, orientation_user, window_size, stride, emg);          
            
            gesture_struct.("features_per_window" + "Win" + window_size + "Stride" + stride) = features_per_window;

            testing{gesture_number-RepTraining} = gesture_struct;

        end

        userInfo = userData.userInfo;
        sync = userData.sync;

        model_name = user_full_dir_out + "userData" + ".mat";
        save(model_name, "userInfo", "sync", "training", "testing");

    end




    t_end = toc(t_begin);
    fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

end
