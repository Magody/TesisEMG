function context = generateContext(params, classes_num_to_name)
        
    context = containers.Map();
    context('classes_num_to_name') = classes_num_to_name;
    
    classes_name_to_num = deepCopyMapInverse(classes_num_to_name);
    
    context('classes_name_to_num') = classes_name_to_num;
    % prepare_environment();    
    
    if isfield(params, "rangeDown")
        context('rangeDownTrain') = params.rangeDown;
    else
        context('rangeDownTrain') = 1;
    end
    
    if isfield(params, "rangeDownValidation")
        context('rangeDownValidation') = params.rangeDownValidation;
    else
        context('rangeDownValidation') = 1;
    end
    
    if isfield(params, "rangeDownTest")
        context('rangeDownTest') = params.rangeDownTest;
    else
        context('rangeDownTest') = 1;
    end
    
    
    
    context('tabulation_mode') = 2;
    context('is_preprocessed') = true;
    context('noGestureDetection') = false;
    context('window_size') = params.window_size;
    context('stride') = params.stride;
    assignin('base','WindowsSize',  params.window_size);
    assignin('base','Stride',  params.stride);
    
    
end

