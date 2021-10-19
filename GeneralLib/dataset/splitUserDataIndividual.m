function [dataset_training, dataset_validation, dataset_testing] = ... 
    splitUserDataIndividual(user_folder, path_to_data, ignoreGestures, ...
    porc_training, porc_validation, mode_shuffle)
    
    userData = loadUserByNameAndDir(user_folder, path_to_data, false);
    
    if mode_shuffle == "packet"
        dataset_part1 = packerByGestures(userData.training, ignoreGestures);
        dataset_part2 = packerByGestures(userData.testing, ignoreGestures);
    else
        dataset_part1 = userData.training;
        dataset_part2 = userData.testing;
    end
    dataset_complete = [dataset_part1, dataset_part2];
    
    dataset_complete_length = length(dataset_complete);
    
    amount_training = floor(dataset_complete_length*porc_training);
    amount_validation = floor(dataset_complete_length*porc_validation);
    amount_testing = dataset_complete_length - amount_training - amount_validation;
    
    next_index = amount_training+1;
    
    dataset_training = dataset_complete(1:amount_training);
    if amount_validation == 0
        dataset_validation = {};
    else
        dataset_validation = dataset_complete(next_index:(amount_training+amount_validation));
        next_index = amount_training+amount_validation+1;
    end
    dataset_testing = dataset_complete(next_index:(next_index+amount_testing-1));
end

