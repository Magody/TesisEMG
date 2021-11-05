function normaliced = standardizationCustom(matrix, mode, data_mean, data_std)
% normalices 2D only

    if data_std == 0
        normaliced = matrix;
        return;
    end

    if mode == "horizontal2D"
        normaliced = (matrix - data_mean) ./ repmat(data_std, [1, size(matrix, 2)]);
    elseif mode == "vertical2D"
        normaliced = (matrix - data_mean) ./ repmat(data_std, [size(matrix, 1), 1]);
    elseif mode == "all"
        normaliced = (matrix - data_mean) / data_std;
    end
end