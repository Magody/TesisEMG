function metrics = getMetricsFromCorrectIncorrect(corrects, incorrects)

metrics = struct();

linear_corrects = corrects';
linear_incorrects = incorrects';
total = linear_corrects(:) + linear_incorrects(:);
metrics.accuracy_by_t = linear_corrects(:) ./ total;

metrics.accuracy = mean(metrics.accuracy_by_t);



end

