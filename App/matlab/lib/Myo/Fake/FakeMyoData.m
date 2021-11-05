classdef FakeMyoData < handle
    properties
        emg_log = [];
        pose_log = [];
        rot_log = [];
        quat_log = [];
        gyro_log = [];
        accel_log = [];
        isStreaming = false;
    end
    
    properties(Access=private)
        timerEmg;
        timerPose;
        timerRot;
        map_gesture_index;
        all_gestures;
        mock_gestures;
        mock_gestures_length;
        index_gesture = 1;
        inter_counter_emg = 1;
        
        class_name_to_numstr;
        map_expected;
        separation_to_class;
        class_num_to_name;
    end
    
    
    methods
        function myoData = FakeMyoData()
            
            % the map gesture can change
            % inside App/matlab/data
            myoData.setUserSource(300);
            
            % Myo works in 200Hz = 5ms, but this simulation works with
            % a sampling collection using batchs of emg points
            period = 0.1; % 1/200;
            
            
            %% Initialize timmers
            
            % EMG timer
            myoData.timerEmg = timer;
            myoData.timerEmg.TimerFcn = @(t,v) myoData.updateEMG();
            myoData.timerEmg.ExecutionMode = 'fixedRate';
            myoData.timerEmg.Period =  period;
            
            % Pose timer
            myoData.timerPose = timer;
            myoData.timerPose.TimerFcn = @(t,v) myoData.updatePose();
            myoData.timerPose.ExecutionMode = 'fixedRate';
            myoData.timerPose.Period = 0.5; % ?
            
            % Rot timer
            myoData.timerRot = timer;
            myoData.timerRot.TimerFcn = @(t,v) myoData.updateRot();
            myoData.timerRot.ExecutionMode = 'fixedRate';
            myoData.timerRot.Period =  period;
            
        end
        
        function setUserSource(myoData, user_id)
            
            path_root = "/home/magody/programming/MATLAB/tesis/";
            
            userData = load(path_root + "Data/preprocessing/user" + user_id + "/userData.mat");
            myoData.all_gestures = userData.training;
            
            order_classes = ["noGesture", "fist", "open", "pinch", "waveIn", "waveOut"];
            order_separation = ["0", "25", "50", "75", "100", "125"];
            
            myoData.class_name_to_numstr = containers.Map(order_classes, order_separation);
            myoData.class_num_to_name = containers.Map([1,2,3,4,5,6], order_classes);

            myoData.separation_to_class = containers.Map(order_separation, ...
                [string(myoData.all_gestures{1}.gestureName), ...
                string(myoData.all_gestures{26}.gestureName), ...
                string(myoData.all_gestures{51}.gestureName), ...
                string(myoData.all_gestures{76}.gestureName), ...
                string(myoData.all_gestures{101}.gestureName), ...
                string(myoData.all_gestures{126}.gestureName)]);
            
            
            class_to_separation = containers.Map(values(myoData.separation_to_class), keys(myoData.separation_to_class));

            myoData.map_expected = containers.Map();
            for index_class=1:length(order_classes)
                class = order_classes(index_class);
                separation = order_separation(index_class);

                myoData.map_expected(separation) = class_to_separation(class);
            end
            
            
            myoData.setSimulationGestures("all");
        end
        
        function setSimulationRandom(myoData)
            type = myoData.class_num_to_name(randi([1 6]));
            myoData.setSimulationGestures(type);
            myoData.index_gesture = randi([1 25]);
        end
        
        function emg = getSample(myoData, type)
            
            assumption_num = myoData.class_name_to_numstr(type);
                
            actual_separation= str2double(myoData.map_expected(assumption_num));
            index_begin = 1 + actual_separation;
            
            gesture = myoData.all_gestures{index_begin};
            emg = gesture.emg;
        end
        
        function setSimulationGestures(myoData, type)
            
            wasStreaming = false;
            if myoData.isStreaming
               wasStreaming = true; 
               myoData.stopStreaming();
            end
            
            if type == "all"
                
                myoData.mock_gestures = myoData.all_gestures(26:150);
            else
                
                assumption_num = myoData.class_name_to_numstr(type);
                
                
                actual_separation= str2double(myoData.map_expected(assumption_num));
                index_begin = 1 + actual_separation;
                index_end = index_begin + 25 - 1;
                myoData.mock_gestures = myoData.all_gestures(index_begin:index_end);
                
            end
            myoData.mock_gestures_length = length(myoData.mock_gestures);
            
            myoData.index_gesture = 1;
            myoData.inter_counter_emg = 1;
            
            if wasStreaming
                wasStreaming.startStreaming();
            end
            
        end
        
        function updateEMG(myoData)
            n_points_to_simulate = 100;
            
            emg = myoData.mock_gestures{myoData.index_gesture}.emg;
            emg_length = size(emg, 1);
            
            index_emg_begin = myoData.inter_counter_emg;
            next_point = index_emg_begin + n_points_to_simulate - 1;
            
            select_next_gesture = false;
            if next_point > emg_length
                select_next_gesture = true;
                index_emg_end = emg_length;
            else
                index_emg_end = next_point;
            end
            
            
            % always the same emg for quick test
            myoData.emg_log = [ myoData.emg_log; emg(index_emg_begin:index_emg_end, :) ];
            
            if select_next_gesture
                myoData.inter_counter_emg = 1; % reset
                
                myoData.index_gesture = mod(myoData.index_gesture+1, myoData.mock_gestures_length);
                if myoData.index_gesture == 0
                    myoData.index_gesture =  myoData.mock_gestures_length;
                end
            else
                myoData.inter_counter_emg = myoData.inter_counter_emg + n_points_to_simulate;
            end
            
        end
        
        function updatePose(myoData)
            myoData.pose_log = [ myoData.pose_log; ceil(rand(1,1)*6) ];
        end
        
        function updateRot(myoData)% updating all, instead
            myoData.rot_log = [ myoData.rot_log; rand(50, 9) ];
            myoData.accel_log = [ myoData.accel_log; rand(50, 3) ];
            myoData.quat_log = [ myoData.quat_log; rand(50, 4) ];
            myoData.gyro_log = [ myoData.gyro_log; rand(50, 3) ];
        end
        
        function startStreaming(myoData)
            myoData.isStreaming = true;
            start(myoData.timerEmg);
            start(myoData.timerPose);
            start(myoData.timerRot);
        end
        
        function stopStreaming(myoData)
            myoData.isStreaming = false;
            stop(myoData.timerEmg);
            stop(myoData.timerPose);
            stop(myoData.timerRot);
        end
        
        function clearLogs(myoData)
            myoData.emg_log = [];
            myoData.pose_log = [];
            myoData.rot_log = [];
            myoData.quat_log = [];
            myoData.gyro_log = [];
            myoData.accel_log = [];
            
        end
    end
end