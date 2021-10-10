function gt_gestures_labels_num = mapGestureLabelsToNumbers(numWindows, ...
    gt_gestures_labels, map_labels_to_num)
%MAPGESTURELABELSTONUMBERS
%{ 
Mapping each word to a number
Creo vector de etiquetas de Ground truth con valores numericos
%}
% gt_gestures_labels: strins[][], the dims are: 1xnumWindows

gt_gestures_labels_num=zeros(numWindows,1);

for i = 1:numWindows
    gt_gestures_labels_num(i,1)= map_labels_to_num(gt_gestures_labels(1,i));
end
            
            
end

