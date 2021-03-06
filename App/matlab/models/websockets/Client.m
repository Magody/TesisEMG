classdef Client < WebSocketClient
    %CLIENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        qnn;
        emg;
        sync;
    end
    
    methods
        function obj = Client(varargin)
            %Constructor
            obj@WebSocketClient(varargin{:});
           
            load('model.mat');

            qnnOption = QNNOption(params.numNeuronsLayers, params.transferFunctions, ...
                            params.lambda, params.learningRate, params.numEpochsToIncreaseMomentum, ...
                            params.momentum, params.initialMomentum, ...
                            params.miniBatchSize, params.gamma, params.epsilon);

            obj.qnn = QNN(qnnOption, params.reserved_space_for_gesture);
            obj.qnn.initTheta(theta);
            obj.qnn.gameReplay = gameReplay;
        end
        
        function sendCustomMessage(this, id, metadata, data)
            id_part = strcat('{"id": ', num2str(id));
            metadata_part = strcat(', "metadata": "', char(metadata));
            data_part = strcat('", "data": "', char(data));
            this.send(strcat(id_part, strcat(metadata_part, strcat(data_part, '"}'))));
        end
    end
    
    methods (Access = protected)
        function onOpen(obj,message)
            % This function simply displays the message received
            fprintf('%s\n',message);
        end
        
        function onTextMessage(this, message)
            % fprintf('Message received:\n%s\n', message);
            msg = jsondecode(jsondecode(message));
            disp(msg);
            
            if(msg.metadata == -2002)
                disp("CORRIGIENDO");
                txt_data = ""+msg.data;
                gestureName = txt_data.replace("string->", "");
                for rep=1:10
                    [summary_episodes, summary_classifications_mode] = this.qnn.train_one(gestureName, this.emg, this.sync, 200, 20);

                end
                fprintf("Win: %d, Loss: %d\n", summary_classifications_mode(1), summary_classifications_mode(2));
    
            end
            
            if(msg.metadata == 1004)
                
                for sim=1:50 % size(this.emg,1)
                
                    this.sendCustomMessage(2001, strcat('int->', "10000"), strcat('matrix->', convertMatrixToExplicitString(this.emg(1:sim, :))));

                end
                
                gesture_prediction = this.qnn.predict(this.emg, this.sync, 200, 20);  % params.window_size, params.stride

                this.sendCustomMessage(2002, strcat('int->', "10000"), strcat('string->', string(gesture_prediction)));

                
            end
            
            
            
            
        end
        
        function onBinaryMessage(obj,bytearray)
            % This function simply displays the message received
            fprintf('Binary message received:\n');
            fprintf('Array length: %d\n',length(bytearray));
        end
        
        function onError(obj,message)
            % This function simply displays the message received
            fprintf('Error: %s\n',message);
        end
        
        function onClose(obj,message)
            % This function simply displays the message received
            fprintf('%s\n',message);
        end
        
        
    end
end

