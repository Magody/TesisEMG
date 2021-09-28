function metrics = getMetricsFromCorrectIncorrect(corrects, incorrects)

metrics = containers.Map();

linear_corrects = corrects';
linear_incorrects = incorrects';
total = linear_corrects(:) + linear_incorrects(:);
metrics('accuracy_by_t') = linear_corrects ./ total;

total_preserving_user = corrects + incorrects;
metrics('accuracy_by_user') = mean(corrects ./ total_preserving_user, 2);

metrics('accuracy') = mean(metrics('accuracy_by_t'));



end

