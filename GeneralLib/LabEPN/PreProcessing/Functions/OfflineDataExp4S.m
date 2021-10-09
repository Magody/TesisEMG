% --------------------------------------------------------------------
% This code is based on the paper entitled                     
% "An Energy-based Method for Orientation Correction
% of EMG Bracelet Sensors in Hand Gesture Recognition Systems"
% by Victor Hugo Vimos T.
%
% *Victor Hugo Vimos T / victor.vimos@epn.edu.ec
% --------------------------------------------------------------------

function emgDataTested = OfflineDataExp4S(timeWindow,stepWindow,correctionMode, verbose_level)


dataPacket      =   evalin('base','dataPacket');
dataPacketSize	=   evalin('base','dataPacketSize');
pathUser        =   evalin('base','pathUser');
pathOrigin      =   evalin('base','pathOrigin');

userIndex       =   evalin('base','userIndex');
emgRepetition   =   evalin('base','emgRepetition');
stepControl     =   evalin('base','stepControl');

is_legacy = evalin('base','is_legacy');


orientation     =   evalin('base','orientation');
Stride = evalin('base', 'Stride');
WindowsSize = evalin('base', 'WindowsSize');
% ==============================================================================
% Step control manages every window step over the EMG signal
%
if stepControl == 1
    
    startWave  = 1;              % First start window point to analize
    endWave    = timeWindow*WindowsSize; % First end window point to analize
    
    assignin('base','startWave', startWave);
    assignin('base','endWave',   endWave);
    
else
    startWave       =   evalin('base', 'startWave');
    endWave         =   evalin('base', 'endWave');
    startWave       =   startWave + stepWindow;   % Next start window point to analize
    endWave         =   endWave   + stepWindow;   % Next end window point to analize
    
    
    assignin('base','startWave',startWave);
    assignin('base','endWave',  endWave);
end
% ==============================================================================


if userIndex~=dataPacketSize+1
    
    
    usuario=dataPacket(userIndex).name;
    assignin('base','usuario',usuario);

    energy_index = find(strcmp(orientation(:,1), usuario));

    assignin('base','low_umbral', orientation{energy_index,3});
    assignin('base','high_umbral',orientation{energy_index,4});

    rand_data=orientation{energy_index,6};        

    recognitionTestUserData=(horzcat(pathOrigin,'/',dataPacket(userIndex).name,'/','userData.mat'));
    load(recognitionTestUserData);
    
    if ~is_legacy
       userData = struct();
       userData.training = training(:);
       userData.testing = testing(:);
       userData.userInfo = userInfo;
       userData.sync = sync;        
    end

    repTraining    = length(userData.training);
    repTesting     = length(userData.testing);
    fields         = fieldnames(userData);

    % =================================================================
    % This csection verifies if orientationCorrection is ON or OFF
    %
    if correctionMode=="off"

    else
          order= orientation{energy_index,2};            
    end

    sequence_mad  = evalin('base','sequence_mad');

    if sequence_mad==true
        sequence_=WM_X(order);
    else

    end

    % ================================================================

    aux__   = fieldnames(userData);
    aux_tr_ = find(aux__=="training");
    aux_ts_ = find(aux__=="testing");

    noGestureDetection  = evalin('base','noGestureDetection');        
    if noGestureDetection == false            
        if  emgRepetition == 176                
            assignin('base','emgRepetition', 201);                
        elseif emgRepetition >= 151 && emgRepetition <= 175                
            emgRepetition=emgRepetition+25;
        end            
    end


    if emgRepetition <= repTraining + repTesting


        % =================================================================
        % This code loads EMG signal fron taining or testing folder
        %
        if rand_data(emgRepetition) <=repTraining

            emgData        = userData.training{rand_data(emgRepetition),1}.emg(:,WM_X(orientation{energy_index,5})); % ROTATION
            emgData        = emgData(:,sequence_);                                                         % CORRECTION  
            fromfield      = fields{aux_tr_,1};
            field_ground   = fieldnames(userData.(fromfield){rand_data(emgRepetition),1});
        else
            emgData        = userData.testing{rand_data(emgRepetition)-repTraining,1}.emg(:,WM_X(orientation{energy_index,5})); % ROTATION
            emgData        = emgData(:,sequence_);                                                                   % CORRECTION
            fromfield      = fields{aux_ts_,1};
            field_ground   = fieldnames(userData.(fromfield){rand_data(emgRepetition)-repTraining,1});
        end

        % =================================================================


        % =================================================================
        % This code sets "start" and  "end" index for every window step
        %
        JP_Longitud=length(emgData);
        assignin('base','JP_Longitud', JP_Longitud);

        if endWave + Stride > length(emgData) %*********************
            %endWave     = length(emgData); %<**********************
            %startWave   = endWave-timeWindow*200; %****************
            controlExit = true;
        else
            controlExit = false;
        end
        assignin('base','controlExit',controlExit);
        % ================================================================

        % ================================================================
        % indexes
        groundTruthLocation  = find(field_ground=="groundTruth");
        gestureNameLocation  = find(field_ground=="gestureName");

        if isempty(groundTruthLocation)
            groundTruth_      = zeros(1,length(emgData));
        else
            if fromfield =="training"
                groundTruth_      = userData.(fromfield){rand_data(emgRepetition),1}.groundTruth;
            else
                groundTruth_      = userData.(fromfield){rand_data(emgRepetition)-repTraining,1}.groundTruth;  % ??????? -repTraining
            end
        end

        assignin('base','groundTruth_',groundTruth_);

        % ===============================================================

        emgDataTested=emgData(startWave:endWave,:);

        timeAnalized   = (endWave-(timeWindow*WindowsSize));
        timeAnalized   = cast(timeAnalized,'double');
        timeAnalized   = timeAnalized/WindowsSize;
        timeAnalized   = timeAnalized+((timeWindow)*(3/5));
        assignin('base','timeAnalized',timeAnalized);

        % =================================================
        % This code plots the RAW EMG and the groundTruth
        %
%         figure(1);
%         subplot(2,1,1);
%         plot(emgData)
%         xlim([0 length(emgData)])
%         ylim([-2 2])
%         grid on
% 
%         subplot(2,1,1)
%         rect = findall(gcf,'Type', 'rectangle');
%         delete(rect);
%         rectangle('Position',[startWave -1  round(timeWindow*WindowsSize) 2],'EdgeColor','r')
% 
%         subplot(2,1,2);
%         plot(groundTruth_)
%         xlim([0 length(emgData)])
%         ylim([-0.5 2])
%         grid on
        % =====================================================


        % =====================================================================================
        % This code shows the information
        %
        if  stepControl<= 1+((length(emgData)-WindowsSize)/Stride) %*********************

            if verbose_level >=1
                disp('                Pseudo Online Test             ')
                disp('                Classifier testing              ')
                fprintf('User number = %d / %d\n',userIndex-2,dataPacketSize-2);
                fprintf('User name = %s\n',usuario);
                fprintf('User channel = %d\n',order);
                fprintf('Current gesture repetition = %d / %d \n',rand_data(emgRepetition),repTraining+repTesting);
                fprintf('Current window = %d \n',stepControl);
                fprintf('Data from field = %s \n',fromfield);
            end

            if fromfield=="training"
                %%%%CAMBIODANNYINICIO%%%%
                if verbose_level >= 1
                    fprintf('Real Training Target is = %s \n',userData.training{rand_data(emgRepetition),1}.gestureName);

                end
                %%%%CAMBIODANNYFIN%%%%

                assignin('base','gestureName_',userData.training{rand_data(emgRepetition),1}.gestureName);

                if userData.training{rand_data(emgRepetition),1}.gestureName ~= "noGesture"
                assignin('base','groundTruthIndex_',userData.training{rand_data(emgRepetition),1}.groundTruthIndex);
                assignin('base','groundTruth_x',userData.training{rand_data(emgRepetition),1}.groundTruth);
                end

            elseif fromfield=="testing"

                if isempty(gestureNameLocation)
                    %%%%CAMBIODANNYINICIO%%%%
                    if verbose_level >= 1
                        fprintf('Real Testing Target is unknown \n');
                    end
                    %%%%CAMBIODANNYFIN%%%%

                    assignin('base','gestureName_','unknown');

                else
                    %%%%CAMBIODANNYINICIO%%%%
                    if verbose_level >= 1
                        fprintf('Real Testing Target is = %s \n',userData.testing{rand_data(emgRepetition)-repTraining,1}.gestureName);

                    end
                    %%%%CAMBIODANNYFIN%%%%

                    assignin('base','gestureName_',userData.testing{rand_data(emgRepetition)-repTraining,1}.gestureName);

                    if userData.testing{rand_data(emgRepetition)-repTraining,1}.gestureName ~= "noGesture"
                    assignin('base','groundTruthIndex_',userData.testing{rand_data(emgRepetition)-repTraining,1}.groundTruthIndex);
                    assignin('base','groundTruth_x',userData.testing{rand_data(emgRepetition)-repTraining,1}.groundTruth);
                   end
                end

            end
         end %****************************

        % ==========================================================================================


        % ==================================================================
        % This section checks the end index of the signal and increases
        % the next emg gestures register to be loaded
        %
        if controlExit==true
            % This code runs when end windows index goes to the last value
            % Window over end limit
            %
            assignin('base','stepControl',1);

            % =====================================================================
            conditionBuffer       = false;
            conditionBuffer_one   = false;
            conditionBuffer_two   = false;

            responseBuffer      = 'noGesture';  % First  buffer gesture located the buffer is "noGesture"
            responseBuffer_one  = 'noGesture';  % Second  buffer gesture located the buffer is "noGesture"
            responseBuffer_two  = 'noGesture';  % Third  buffer gesture located the buffer is "noGesture"

            assignin('base','conditionBuffer',     conditionBuffer);
            assignin('base','conditionBuffer_one', conditionBuffer_one);
            assignin('base','conditionBuffer_two', conditionBuffer_two);

            assignin('base','responseBuffer',     responseBuffer);
            assignin('base','responseBuffer_one', responseBuffer_one);
            assignin('base','responseBuffer_two', responseBuffer_two);


        else
            % This code runs when end windows index does not go to the last value
            % Window end limit is not over
            %
            assignin('base','stepControl',stepControl+1);
        end
        % ==================================================================

    else

        assignin('base','userIndex',userIndex+1);

        emgRepetition = evalin('base', 'rangeDown');            
        assignin('base','emgRepetition',emgRepetition);

        assignin('base','controlExit',false);
        assignin('base','usuario','NaN');
        assignin('base','change_user',false);

        emgDataTested=zeros(10,8);
    end
else
    
    assignin('base','testControl',false);
    emgDataTested=zeros(10,8);
    
end

end

