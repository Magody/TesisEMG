function prepare_environment(data_dir, verbose_level)

if nargin== 1
   verbose_level = 1; 
end

on  = true;
off = false;

%Conversion de JSON a .mat (si es necesario)
% root_        = pwd;
data_gtr_dir = horzcat(data_dir,'General/training');
data_gts_dir = horzcat(data_dir,'General/testing');
data_sts_dir = horzcat(data_dir,'Specific');

if length(dir(data_gtr_dir))>2 || length(dir(data_gts_dir))>2 || length(dir(data_sts_dir))>2
    % No Data conversion
    if verbose_level >= 1
        disp('Data conversion already done');
    end
else
    % Data conversion needed
    jsontomat;
end


%==============Parameters for Code_0 (preprocesser of emg data)==========

assignin('base','post_processing',     on);   %on si quiero post procesamiento en vector de etiquetas resultadnte                                          %off si quiero solo recomp -10 x recog 
 
% if randomGestures is on, all will be random and packet EMG will not
% put the gestures one after other secuentially
assignin('base','randomGestures',     on);   %on si quiero leer datos randomicamente
assignin('base','noGestureDetection', on);  %off si no quiero considerar muestras con nogesture - OJO> actualmente el gesto predicho es la moda sin incluir no gesto
%limite superior de rango de muestras a leer
assignin('base','rangeValues', 150);  %up to 300 - rango de muestras PERMITIDO que uso dentro del dataset, del cual tomo "RepTraining" muestras
% if true: locates secuentially the gestures like: 
%   (nogestures if actived), fist, open, pinch, wave in, wave out
%   (nogestures if actived), fist, open, pinch, wave in, wave out, etc


end

