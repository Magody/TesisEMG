function packet_gestures = packerByGestures(gestures, ignore_gesture)

    classes_separation = containers.Map(["0", "25", "50", "75", "100", "125"], ["noGesture", "fist", "open", "pinch", "waveIn","waveOut"]);
    

    if nargin == 1
       ignore_gesture = ""; 
    end

    len_gestures = length(gestures);
    
    if ignore_gesture ~= ""
        len_gestures = len_gestures - 25;
    end
    packet_gestures = cell([1, len_gestures]);
        
    index_packet = 1;
    for x=1:25
        
        for separation=0:25:125
            if classes_separation(num2str(separation)) ~= ignore_gesture
                packet_gestures(index_packet) = gestures(x+separation);
                index_packet = index_packet + 1;
            end
        end

    end

end

