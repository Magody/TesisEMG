function gt_gestures_labels_num = mapGestureLabelsToNumbers(numWindows, gt_gestures_labels)
%MAPGESTURELABELSTONUMBERS
%{ 
Mapping each word to a number
Creo vector de etiquetas de Ground truth con valores numericos
%}
% gt_gestures_labels: strins[][], the dims are: 1xnumWindows

gt_gestures_labels_num=zeros(numWindows,1);

for i = 1:numWindows
    if gt_gestures_labels(1,i) == "waveOut"
        gt_gestures_labels_num(i,1)=1;
    elseif gt_gestures_labels(1,i) == "waveIn"
        gt_gestures_labels_num(i,1)=2;        %CAMBIAR
    elseif gt_gestures_labels(1,i) == "fist"
        gt_gestures_labels_num(i,1)=3 ;         %CAMBIAR
    elseif gt_gestures_labels(1,i) == "open"
        gt_gestures_labels_num(i,1)=4;         %CAMBIAR
    elseif gt_gestures_labels(1,i) == "pinch"
        gt_gestures_labels_num(i,1)=5;       %CAMBIAR
    elseif gt_gestures_labels(1,i) == "noGesture"
        gt_gestures_labels_num(i,1)=6;        %CAMBIAR
    end
end
            
            
end

