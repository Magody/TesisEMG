function matrix_string = convertMatrixToExplicitString(matrix)

matrix_string = "[";
m = size(matrix, 1);

for i=1:m
    if(i < m)
        matrix_string = strcat(matrix_string, strcat(convertArrayToExplicitString(matrix(i, :)), ",") );
    else
        matrix_string = strcat(matrix_string, strcat(convertArrayToExplicitString(matrix(i, :)), "]") );
    
    end
end

end

