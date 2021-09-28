function matrix_without_nan = getRowsNotNan(matrix_sparse_nan)
%GETROWSNOTNAN


matrix_without_nan = [];

for i=1:size(matrix_sparse_nan, 1)
    if mean(isnan(matrix_sparse_nan(i,:))) == 0
        matrix_without_nan = [matrix_without_nan; matrix_sparse_nan(i,:)]; 
    end
end

end

