function gt_gestures_labels = mapGroundTruthToLabels(numWindows, emg_window_size, stride, groundTruthIndex_GT, gestureName_GT, part_of_ground_truth_to_identify)
%MAPGROUNDTRUTHTOLABELS 
%Creo vector de etiquetas para Ground truth x ventana
% if the window is holding a EMG_window_size/5 part at least of the groundtruth, is labeled as the gesture.

if nargin < 6
    part_of_ground_truth_to_identify = 0.2;
end

%Creo vector con limite derecho de cada ventana. Ejm: win=200,stride=30 -> 200 230 260.... 980
gt_gestures_pts=zeros(1, numWindows); %% CORREGIR
gt_gestures_pts(1,1)=emg_window_size;
for k = 1:numWindows-1
    gt_gestures_pts(1,k+1)=gt_gestures_pts(1,k)+stride;
end
% assignin('base','gt_gestures_pts',gt_gestures_pts); DELETE?
            
gt_gestures_labels=strings;

threshold = ceil((groundTruthIndex_GT(1,2) - groundTruthIndex_GT(1,1))*part_of_ground_truth_to_identify);

for window = 1:numWindows
    window_right_limit = gt_gestures_pts(1,window);
    window_left_limit = window_right_limit - emg_window_size;
    
    if (window_right_limit >= groundTruthIndex_GT(1,1) + threshold && ...
       window_right_limit <= groundTruthIndex_GT(1,2)) || ...
       (window_left_limit <= groundTruthIndex_GT(1,1) && ...  % contains
       window_right_limit >= groundTruthIndex_GT(1,2))  || ...
       (window_left_limit >= groundTruthIndex_GT(1,1) && ...
       window_left_limit <= groundTruthIndex_GT(1,2) - threshold)
       % the end of the window is inside the ground truth
       
       gt_gestures_labels(1,window)=string(gestureName_GT);
    else
       % the end of the window is outside the ground truth: left or right
       gt_gestures_labels(1,window)="noGesture";
    end
end

end

