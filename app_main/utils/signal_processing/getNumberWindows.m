function total_windows_in_emg = getNumberWindows(emg_points, window_size, stride, include_residue)
%GETNUMBERWINDOWS this function calculate the number of windows in a emg
%signal considering the shift of stride
%{
emg_points: number, number of point usually around 996
window_size: number, usually around 200
stride: number, the separation between windows usually around 20
include_residue: boolean, if true the last window will be considered even
                          if its size is less than window_size

%}
total_jumps_for_end_index = (emg_points-window_size)/stride;
if include_residue
    total_windows_in_emg = ceil(total_jumps_for_end_index)+1;  %+1 is due to the first window
else
    total_windows_in_emg = floor(total_jumps_for_end_index)+1;  %+1 is due to the first window
end

end

