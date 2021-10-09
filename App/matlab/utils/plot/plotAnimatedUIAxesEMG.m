function plotAnimatedUIAxesEMG(~, ~) % timerObject, timerInfo

global deviceType myoObject UIAxes h plot_call_counter sample;


% Sample with last 1000 points
% sample = struct('emg', [], 'quaternions', [], 'gyro', [],'accel');


plot_signal_length = 100;
plot_offset = max(0, plot_call_counter-plot_signal_length);

if deviceType == DeviceName.myo
    begin_x = 0+plot_offset;
    end_x = plot_signal_length+plot_offset;
    % UIAxes.XLim = [begin_x, end_x];
    % UIAxes.XTick = begin_x:10:end_x;
    
    plot_call_counter = plot_call_counter + 1;
    addpoints(h, plot_call_counter, myoObject.myoData.emg_log(end, 1));
    drawnow
end


end

