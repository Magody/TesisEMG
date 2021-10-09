function array_string = convertArrayToExplicitString(array)

array_string = "[";

for i=1:numel(array)
    
    if(i < numel(array))
        array_string = strcat(array_string, strcat(num2str(array(i)), ", "));
    else
        array_string = strcat(array_string, strcat(num2str(array(i)), "]"));
    end
    
    
end

end

