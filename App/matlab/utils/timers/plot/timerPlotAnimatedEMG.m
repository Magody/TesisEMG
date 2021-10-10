function timerPlotAnimatedEMG(~, ~) % timerObject, timerInfo
global deviceType myoObject h plot_call_counter sample;

% Sample with last 1000 points
% sample = struct('emg', [], 'quaternions', [], 'gyro', [],'accel');


plot_signal_length = 20;
plot_offset = max(0, plot_call_counter-plot_signal_length);

if deviceType == DeviceName.myo
    
    axis([0+plot_offset plot_signal_length+plot_offset -1.3 1.3])
    
    plot_call_counter = plot_call_counter + 1;
    addpoints(h, plot_call_counter, myoObject.myoData.emg_log(end, 1));
    drawnow
end


end

