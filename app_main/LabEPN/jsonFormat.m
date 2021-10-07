function jsonFormat(jsonName, version, Rep, results)


txt = jsonencode(results);
fid = fopen(jsonName, 'wt');
fprintf(fid,txt);
fclose(fid);


end
