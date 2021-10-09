function prepare_environment(environment_options)


if nargin < 1
   on = true;
    off = false;
    environment_options = struct();
    environment_options.post_processing = on;
    environment_options.randomGestures = off;
    environment_options.noGestureDetection = on;
    environment_options.rangeValues = 150;
    environment_options.packetEMG = true;
end


assignin('base','post_processing', environment_options.post_processing);   %on si quiero post procesamiento en vector de etiquetas resultadnte                                          %off si quiero solo recomp -10 x recog 
 
% if randomGestures is on, all will be random and packet EMG will not
% put the gestures one after other secuentially
assignin('base','randomGestures',     environment_options.randomGestures);   %on si quiero leer datos randomicamente
assignin('base','noGestureDetection', environment_options.noGestureDetection);  %off si no quiero considerar muestras con nogesture - OJO> actualmente el gesto predicho es la moda sin incluir no gesto
%limite superior de rango de muestras a leer
assignin('base','rangeValues', environment_options.rangeValues);  %up to 300 - rango de muestras PERMITIDO que uso dentro del dataset, del cual tomo "RepTraining" muestras
% if true: locates secuentially the gestures like: 
%   (nogestures if actived), fist, open, pinch, wave in, wave out
%   (nogestures if actived), fist, open, pinch, wave in, wave out, etc
assignin('base','packetEMG',     environment_options.packetEMG);

end

