function [Numero_Ventanas,EMG,Vector_EMG_Features,Vector_EMG_Tiempos,Vector_EMG_Puntos,Nombre_Usuario,gestureName_,groundTruthIndex_,groundTruth_x] = ...
    Code_1(orientation,dataPacketSize,RepTraining, verbose_level)
%%%%CAMBIODANNYINICIO%%%%
if nargin<5
  verbose_level = true;
end
%%%%CAMBIODANNYFIN%%%%

 Numero_Ventanas      = 0;
 EMG                  = 0;
 Vector_EMG_Features  = 0;
 Vector_EMG_Tiempos   = 0;
 Vector_EMG_Puntos    = 0 ;
 Nombre_Usuario       = 'NaN';
 gestureName_         = 'NaN';
 groundTruthIndex_    = [0 1];
 groundTruth_x        = logical(zeros(1,1000));


testControl = evalin('base', 'testControl');
Stride = evalin('base', 'Stride');
WindowsSize = evalin('base', 'WindowsSize');


if testControl==true
    
    change_user = evalin('base', 'change_user');
    userIndex   = evalin('base', 'userIndex');
    
    if userIndex>=3 && change_user==false
        
        index_user = evalin('base', 'index_user');
        
        if index_user > dataPacketSize-2
            usuario           = orientation{index_user-1,1};
        else
            usuario           = orientation{index_user,1};
        end
        index_user        = index_user+1;
        assignin('base','change_user',true);
        
    end
    
    emgDataTested     = OfflineDataExp4S(1,Stride,"on", verbose_level-1);
    emgFeatures       = getFeatures(emgDataTested);
    controlExit       = evalin('base','controlExit');
    usuario           = evalin('base','usuario');
    emgRepetition     = evalin('base','emgRepetition');   
    
    if controlExit==false && usuario~="NaN"
        
        timeAnalized            = evalin('base','timeAnalized');
        WindowRep               = evalin('base','JP_Longitud');
        
        
        Numero_Ventanas     = floor((WindowRep - WindowsSize)/Stride); % *******************************************
        EMG                 = emgDataTested;                % EMG de 200 puntos
        Vector_EMG_Features = emgFeatures;                  % Vectores en formato de tabla
       %Vector_EMG_Features = table2array(emgFeatures);     % Vectores en formato de matriz
        Vector_EMG_Tiempos  = timeAnalized;
        Vector_EMG_Puntos   = timeAnalized*200;
        Nombre_Usuario      = usuario;
        
        gestureName_         = evalin('base','gestureName_');
        
        if gestureName_~= "noGesture" && gestureName_ ~= "unknown"
        groundTruthIndex_    = evalin('base','groundTruthIndex_');
        groundTruth_x        = evalin('base','groundTruth_x');
        end
        
             
    elseif controlExit==true && usuario~="NaN"
        
        timeAnalized            = evalin('base','timeAnalized');
        WindowRep               = evalin('base','JP_Longitud');
        
        
        Numero_Ventanas     = floor((WindowRep - WindowsSize)/Stride); % *******************************************
        EMG                 = emgDataTested;                % EMG de 200 puntos
        Vector_EMG_Features = emgFeatures;                  % Vectores en formato de tabla
       %Vector_EMG_Features = table2array(emgFeatures);     % Vectores en formato de matriz
        Vector_EMG_Tiempos  = timeAnalized;
        Vector_EMG_Puntos   = timeAnalized*200;
        Nombre_Usuario      = usuario;
        
        gestureName_         = evalin('base','gestureName_');
        
        if gestureName_~= "noGesture" && gestureName_ ~= "unknown"
        groundTruthIndex_    = evalin('base','groundTruthIndex_');
        groundTruth_x        = evalin('base','groundTruth_x');
        end
        
        
        %clc
        assignin('base','emgRepetition',emgRepetition+1);        
        counter = evalin('base','counter');
        counter=counter+1;
        assignin('base','counter',counter);
      
        
        if counter<RepTraining+1
            %clc
            %disp('Fin de la muestra, 40 ventanas')
            %pause(2)

        else
            
            %assignin('base','emgRepetition',301);
            counter=301;
            assignin('base','counter',301);
                 
        end
        
        if counter==301
            %clc
            %disp('Fin de la muestra, 40 ventanas')
            %disp('Cambio de usuario')
            assignin('base','counter',1); 
            %pause(2)
        end
     
    end
        
    testControl    = evalin('base','testControl');
end

end

