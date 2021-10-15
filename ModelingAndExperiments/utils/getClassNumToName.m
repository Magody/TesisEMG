function classes_num_to_name = getClassNumToName(gestures_list, ignoreGestures)
    
    gestures_list_length = length(gestures_list);
    output_neurons = gestures_list_length;

    ignoreGesturesForTest = [];
    reduce = false;
    reduction = 0;
    for i=1:length(ignoreGestures)
        ignore_gesture = ignoreGestures(i);
        if ignore_gesture ~= "" && ignore_gesture ~= "noGesture"
           reduce = true;
           reduction = reduction + 1;
           ignoreGesturesForTest = [ignoreGesturesForTest, ignore_gesture];
        end
    end

    if reduce
        output_neurons = output_neurons - reduction;
        gestures_list_reduced = strings([1, gestures_list_length-reduction]);

        index_gesture_reduced = 1;
        for index_gesture=1:gestures_list_length
            gesture_string = gestures_list(index_gesture);

            ignore_gesture_string = false;
            for i=1:length(ignoreGestures)
                if ignoreGestures(i) ~= "" && ignoreGestures(i) ~= "noGesture"
                   if ignoreGestures(i) == gesture_string
                       ignore_gesture_string = true; break;
                   end
                end
            end

            if ~ignore_gesture_string
                gestures_list_reduced(index_gesture_reduced) = gesture_string;
                index_gesture_reduced = index_gesture_reduced + 1;
            end
        end

        classes_num_to_name = containers.Map(1:length(gestures_list_reduced), gestures_list_reduced);

    else
        classes_num_to_name = containers.Map([1, 2, 3, 4, 5, 6], gestures_list);
    end
end

