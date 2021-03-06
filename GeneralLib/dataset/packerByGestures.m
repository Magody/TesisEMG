function packet_gestures = packerByGestures(gestures, ignore_gestures)

    if ~isfield(gestures{1}, 'gestureName')
        disp("\nError, the gestures haven't been labeled in gestureName, returning same without ordering");
        packet_gestures = gestures;
        return;
    end

    order_classes = ["noGesture", "fist", "open", "pinch", "waveIn", "waveOut"];
    
    order_separation = ["0", "25", "50", "75", "100", "125"];
    
    separation_to_class = containers.Map(order_separation, ...
        [string(gestures{1}.gestureName), ...
        string(gestures{26}.gestureName), ...
        string(gestures{51}.gestureName), ...
        string(gestures{76}.gestureName), ...
        string(gestures{101}.gestureName), ...
        string(gestures{126}.gestureName)]);
    class_to_separation = containers.Map(values(separation_to_class), keys(separation_to_class));
    
    map_expected = containers.Map();
    for index_class=1:length(order_classes)
        class = order_classes(index_class);
        separation = order_separation(index_class);
        
        map_expected(separation) = class_to_separation(class);
    end
    
    
    % order in most users (not all): ["noGesture", "fist", "open", "pinch", "waveIn", "waveOut"] 
    

    if nargin == 1
       ignore_gestures = []; 
    end

    len_gestures = length(gestures);
    
    
    if ~isempty(ignore_gestures)
        len_gestures = len_gestures - (25 * length(ignore_gestures));
    end
    packet_gestures = cell([1, len_gestures]);
        
    index_packet = 1;
    for x=1:25
        
        for separation=0:25:125
            actual_separation_string = map_expected(num2str(separation));
            actual_separation_num = str2double(actual_separation_string);
            
            actual_gesture = separation_to_class(actual_separation_string);
            
            
            if searchStringInArray(ignore_gestures, actual_gesture) == -1
                % if not found in ignores, add
                packet_gestures(index_packet) = gestures(x+actual_separation_num);
                index_packet = index_packet + 1;
            end
        end

    end

end

