function groundTruth_GT = getGroundTruthGT(numberPointsEmgData, groundTruthIndex)

groundTruth_GT = zeros(1, numberPointsEmgData);

if groundTruthIndex(1) == 0 && groundTruthIndex(2) == 0
    return;
end

for i=groundTruthIndex(1):groundTruthIndex(2)
    
    groundTruth_GT(1, i) = 1;
end

end

