function index = searchStringInArray(array, search)
index = -1;

for i=1:length(array)
    
    if array(i) == search
       index = i;
       break;
    end

end

