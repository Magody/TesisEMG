function context = generateContext(params, classes_num_to_name)
        
    context = containers.Map();
    context('classes_num_to_name') = classes_num_to_name;
    
    classes_name_to_num = deepCopyMapInverse(classes_num_to_name);
    
    context('classes_name_to_num') = classes_name_to_num;
    % prepare_environment();    
    
    
    context('tabulation_mode') = 2;
    context('is_preprocessed') = true;
    context('noGestureDetection') = false;
    context('window_size') = params.window_size;
    context('stride') = params.stride;
    assignin('base','WindowsSize',  params.window_size);
    assignin('base','Stride',  params.stride);
    
    context("part_of_ground_truth_to_identify") = 0.2;
    
end

