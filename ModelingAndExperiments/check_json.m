fname1 = '/home/magody/Documents/responses.json';
fname2 = '/home/magody/Documents/responses0.json'; 
fname3 = '/home/magody/Documents/responses1.json'; 


json1 = readJSON(fname1);
size1 = whos('json1').bytes/(1024*1024);
fprintf("%.2f [MB]\n", size1);
jsonFormat("1.json", json1);

json2 = readJSON(fname2);
size2 = whos('json2').bytes/(1024*1024);
fprintf("%.2f [MB]\n", size2);
jsonFormat("2.json", json2);





