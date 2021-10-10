function timerPlotAnimatedEMG(~, ~) % timerObject, timerInfo
global deviceType myoObject h plot_index_begin plot_index_end;

% Sample with last 1000 points
% sample = struct('emg', [], 'quaternions', [], 'gyro', [],'accel');

plot_width = 1000;

index_emg_log_end = size(myoObject.myoData.emg_log, 1);

if plot_index_begin >= index_emg_log_end
    return; % not enought data yet
end


plot_offset = max(0, index_emg_log_end-plot_width);

range = plot_index_begin:index_emg_log_end;

if deviceType == DeviceName.myo
    
    axis([0+plot_offset plot_width+plot_offset -1.3 1.3])
    
    for channel=1:8
        addpoints(h, range, myoObject.myoData.emg_log(range, channel));
    end
    drawnow
end

plot_index_begin = index_emg_log_end + 1;

end

