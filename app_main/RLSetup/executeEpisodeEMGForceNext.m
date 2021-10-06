function history_episode = executeEpisodeEMGForceNext(q_neural_network, episode, is_test, functionGetReward, context, verbose_level) 
    
    window_size = context('window_size');
    stride = context('stride');
    gestureName = context('gestureName');
    ground_truth_index = context('ground_truth_index');
    interval_for_learning = context('interval_for_learning'); 
    noGestureDetection  = context('noGestureDetection');
    classes_name_to_num = containers.Map(["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"], {1, 2, 3, 4, 5, 6});
    classes_num_to_name = containers.Map([1, 2, 3, 4, 5, 6], ["waveOut", "waveIn", "fist", "open", "pinch", "noGesture"]);
    
    real_class = classes_name_to_num(string(gestureName));
    
    is_preprocessed = context('is_preprocessed');
    
    if is_preprocessed
        features_per_window = context('features_per_window');
        num_windows = context('num_windows');
        emg_points = context('emg_points');
    else
        emg = context('emg');
        emg_points = size(emg, 1);
        emg_channels = size(emg, 2);
        emg = reshape(emg, [1, emg_points, emg_channels]);
        num_windows = getNumberWindows(emg_points, window_size, stride, false);
    end
    ground_truth_gt = getGroundTruthGT(emg_points, ground_truth_index);

    gt_gestures_pts=zeros([1,num_windows]);
    gt_gestures_pts(1,1)=window_size;

    for k = 1:num_windows-1
        gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
    end
    
    gt_gestures_labels = mapGroundTruthToLabelsWithPts(gt_gestures_pts, ground_truth_index, gestureName, 0.2);
    gt_gestures_labels_num = mapGestureLabelsToNumbers(num_windows, gt_gestures_labels);

    
    history_episode = containers.Map();
    update_counter = 0;
    update_costs = [];
    reward_cummulated = 0;
    
    classification_window_correct = 0;
    classification_window_incorrect = 0;
    
    
    step_t = 0;
    t_for_learning = interval_for_learning; % floor(num_windows/interval_for_learning);
    
    etiquetas_labels_predichas_vector = strings([num_windows-1, 1]);
    
    etiquetas_labels_predichas_vector_simplif=strings();
    predictions = [];
    ProcessingTimes_vector=[];
    TimePoints_vector=[];
    
    n1=0;
    
    tic;
    index_state_begin = 1;
    index_state_end = window_size;
    
    
    % slice = emg(:, index_state_begin:index_state_end, :);slice(:);
    if is_preprocessed
        state = features_per_window(1, :)';
    else
        state = emg(:, index_state_begin:index_state_end, :);
    end

    index_state_begin = index_state_begin + stride;
    index_state_end = index_state_end + stride;
    
    step_t = step_t + 1;
    
    for window=2:num_windows
        
        step_t = step_t + 1;
        
        
        [Qval, action] = q_neural_network.selectAction(state, is_test); %#ok<ASGLU>
        
        % store all predictions
        etiquetas_labels_predichas_vector(window, 1) = classes_num_to_name(action);
    
        if action ~= 6 || real_class == 6
            % if real class is noGesture, we must save the predictions
            predictions = [predictions, action];            
        end

        context('real_action') = gt_gestures_labels_num(window);
        % context is modified by reference
        [reward, ~, ~] = functionGetReward(state, action, context);
        % in this case, new_state is useless
        finish = window == num_windows-1;
        
        if is_preprocessed
            new_state = features_per_window(window, :)';
        else
            new_state = emg(:, index_state_begin:index_state_end, :);
        end
        
        if reward > 0
            classification_window_correct = classification_window_correct + 1;
        else
            classification_window_incorrect = classification_window_incorrect + 1;
        end
        
        if ~is_test
            q_neural_network.saveExperienceReplay(state, action, reward, new_state, finish);
            
            if mod(step_t, t_for_learning) == 0
                % update in each step could be very brute
                history_learning = q_neural_network.learnFromExperienceReplay(episode, verbose_level);
                if history_learning('learned')
                    update_counter = update_counter + 1;
                    update_costs(1, update_counter) = history_learning('mean_cost');
                end
            end
            
        end

        reward_cummulated = reward_cummulated + reward;

        index_state_begin = index_state_begin + stride;
        index_state_end = index_state_end + stride;
        
        % state = new_state;
        
        %Acondicionar vectores - si el signo anterior no es igual al signo acual entocnes mido tiempo
        if window>1 && window~=window-1 && ...   %window_n~=maxWindowsAllowed
                etiquetas_labels_predichas_vector(window,1) ~= etiquetas_labels_predichas_vector(window-1,1)

            n1=n1+1;
            ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
            tic;

            %obtengo solo etiqueta que se ha venido repetiendo hasta instante window_n-1
            etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(window-1,1);

            %obtengo nuevo dato para vector de tiempos
            TimePoints_vector(1,n1)=stride*window+window_size/2;           %necesito dato de stride y tamaño de ventana de Victor

        elseif window== num_windows-1 %==maxWindowsAllowed    % si proceso la ultima ventana de la muestra de señal EMG

            %disp('final window')

            n1=n1+1;
            ProcessingTimes_vector(1,n1) = toc;  %mido tiempo transcurrido desde ultimo cambio de gesto
            tic;

            %obtengo solo etiqueta que no se ha repetido hasta instante window_n-1
            etiquetas_labels_predichas_vector_simplif(1,n1)=etiquetas_labels_predichas_vector(window,1);

            %obtengo dato final para vector de tiempos
            kj=size(ground_truth_gt);  %  se supone q son 1000 puntos
            TimePoints_vector(1,n1)=  kj(1,2);                 %AQUI CAMBIAR  %1000 puntos

        end
        
        state = new_state;

    end
    
    
    
    % class_result_num = mode(predictions);
    etiquetas_labels_predichas_vector_without_NoGesture = strings();
    index_no_gesture = 1;
    for i=1:length(etiquetas_labels_predichas_vector)
        if etiquetas_labels_predichas_vector(i,1)~="noGesture"
            etiquetas_labels_predichas_vector_without_NoGesture(index_no_gesture,1)=etiquetas_labels_predichas_vector(i,1);
            index_no_gesture = index_no_gesture + 1;
        end
    end
    
    if noGestureDetection || string(gestureName) == "noGesture"
        class_result = mode(categorical(etiquetas_labels_predichas_vector));
    else
        class_result = mode(categorical(etiquetas_labels_predichas_vector_without_NoGesture));
    end

    %---  POST - Processing: elimino etiquetas espuria
    post_processing_result_vector_lables=etiquetas_labels_predichas_vector_simplif;
    dim_vect=size(etiquetas_labels_predichas_vector_simplif);
    for i=1:dim_vect(1,2)
        if etiquetas_labels_predichas_vector_simplif(1,i) ~= class_result && etiquetas_labels_predichas_vector_simplif(1,i) ~= "noGesture"
            post_processing_result_vector_lables(1,i)=class_result;
        else
        end  
    end

    % GROUND TRUTH (no depende del modelo)------------
    repInfo.gestureName =  gestureName;
    repInfo.groundTruth = ground_truth_gt;

    response.vectorOfLabels = categorical(post_processing_result_vector_lables);
    
    if length(response.vectorOfLabels) == 1
        response.vectorOfLabels = etiquetas_labels_predichas_vector;
    end
    
    
    response.vectorOfTimePoints = TimePoints_vector; % OK -----  [40 200 400 600 800 999];
    % tiempo de procesamiento
    response.vectorOfProcessingTimes = ProcessingTimes_vector; % OK -----[0.1 0.1 0.1 0.1 0.1 0.1]; % ProcessingTimes_vector'; % [0.1 0.1 0.1 0.1 0.1 0.1]; % 1xw double                                    %CAMBIAR
    response.class =  categorical(class_result); % OK ----- categorical({'waveIn'});                %aqui tengo que usar la moda probablemente           %CAMBIAR


    %-----------------------------------------------

    try
        r1 = evalRecognition(repInfo, response);
    catch
        % warning('EL vector de predicciones esta compuesto por una misma etiqueta -> Func Eval Recog no funciona');
        
        r1.recogResult=0;

    end
    
    recognition = 0;
    
    if ~isempty(r1.recogResult)

        if  r1.recogResult==1
            recognition = 1;
        end
        
    elseif gestureName == "noGesture"
        recognition = r1.classResult;
    end

    if verbose_level > 0
       fprintf("%s: %s, %d\n", string(gestureName), string(class_result), recognition); 
    end

    history_episode('reward_cummulated') = reward_cummulated;
    history_episode('update_costs') = update_costs;
    
    history_episode('classification_window_correct') = classification_window_correct;
    history_episode('classification_window_incorrect') = classification_window_incorrect;
    
    %{
    if (mode(predictions) == real_class) ~= (string(class_result) == string(gestureName))
        disp("Incongruency");
    end
    %}
    is_correct_class = string(class_result) == string(gestureName);
    history_episode('classification_class_correct') = is_correct_class;
    history_episode('classification_class_incorrect') = ~is_correct_class;
    
    
    is_correct_recognition = recognition == 1;
    history_episode('classification_recognition_correct') = is_correct_recognition;
    history_episode('classification_recognition_incorrect') = ~is_correct_recognition;
    
end