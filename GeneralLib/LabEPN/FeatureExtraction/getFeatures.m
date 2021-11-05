function emgActivity = getFeatures(emgData, default, low_umbral, high_umbral)

    TestedData=emgData';
    energy_          = WMoos_F5(TestedData);

    if sum(energy_(1:4,:)) > high_umbral  || sum(energy_(5:8,:)) > low_umbral
        
        var_names_f5 = {'WMoos_F5_Ms1','WMoos_F5_Ms2','WMoos_F5_Ms3','WMoos_F5_Ms4','WMoos_F5_Ms5','WMoos_F5_Ms6','WMoos_F5_Ms7','WMoos_F5_Ms8'};
        WM_F5 = table(energy_(1),energy_(2),energy_(3),energy_(4),energy_(5),energy_(6),energy_(7),energy_(8),'VariableNames', var_names_f5);
        
        Ms1=TestedData(1,:);
        Ms2=TestedData(2,:);
        Ms3=TestedData(3,:);
        Ms4=TestedData(4,:);
        Ms5=TestedData(5,:);
        Ms6=TestedData(6,:);
        Ms7=TestedData(7,:);
        Ms8=TestedData(8,:);
        TableEmgData = table(Ms1,Ms2,Ms3,Ms4,Ms5,Ms6,Ms7,Ms8);

        WM_F1   =   varfun(@WMoos_F1,     TableEmgData);
        WM_F2   =   varfun(@WMoos_F2,     TableEmgData);
        WM_F3   =   varfun(@WMoos_F3,     TableEmgData);
        WM_F4   =   varfun(@WMoos_F4,     TableEmgData);
        % WM_F5   =   varfun(@WMoos_F5,     TableEmgData);


        emgActivity      = [WM_F1,  WM_F2,  WM_F3,  WM_F4,  WM_F5];

    else 
        emgActivity = default;    
    end

end

