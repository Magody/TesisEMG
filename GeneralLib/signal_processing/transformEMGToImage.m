function signal_matrix = transformEMGToImage(emg, height, width, join_all)
    % height should be pair
    signal_length = size(emg, 1);
    channels = size(emg, 2);
    range = ceil(height/2);

    if join_all
        signal_matrix = zeros([2*range, width]);
    else
        signal_matrix = zeros([2*range, width, channels]);
    end
    signal_round = round(emg*(range-1)) + (range+1);

    for channel=1:channels
        for i=1:signal_length
            if join_all
                signal_matrix(signal_round(i, channel), i) = 1;
            else
                signal_matrix(signal_round(i, channel), i, channel) = 1;
            end
        end
    end
end

