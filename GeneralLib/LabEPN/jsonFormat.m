function jsonFormat(jsonName, results)


txt = jsonencode(results);
fid = fopen(jsonName, 'wt');
fprintf(fid,txt);
fclose(fid);




end
