function [history_episodes_by_epoch, summary, t_end] = ...
            trainAndValidate(path_to_framework, path_root, ...
            q_neural_network, general_epochs, do_validation, context, verbose_level)
    %% Libs
    addpath(genpath(path_to_framework));
    addpath(genpath(path_root + 'GeneralLib'));   
    addpath(genpath(path_root + 'ModelingAndExperiments/RLSetup'));  
    addpath(path_root + 'ModelingAndExperiments/Experiments');      


    %% Train
    
    history_episodes_by_epoch = cell([general_epochs, 2]);
    summary = cell([general_epochs, 2]);
    
    t_begin = tic;
    print_step = ceil(general_epochs/2);
    
    for epoch=1:general_epochs
        context('offset_user') = 0;
        context('global_epoch') = epoch;
        % Train
        history_episodes_by_epoch{epoch, 1} = q_neural_network.runEpisodes(@getRewardEMG, 1, context, verbose_level-1);
        

        [classification_window_train, classification_train, recognition_train] = Experiment.getEpisodesEMGMetrics(history_episodes_by_epoch{epoch, 1});
        
        summary{epoch, 1} = struct("classification_window_train", classification_window_train, ...
                                   "classification_train", classification_train, ...
                                   "recognition_train", recognition_train);
                               
        if do_validation
            % validation
            history_episodes_by_epoch{epoch, 2} = q_neural_network.runEpisodes(@getRewardEMG, 2, context, verbose_level-1);

            [classification_window_validation, classification_validation, recognition_validation] = Experiment.getEpisodesEMGMetrics(history_episodes_by_epoch{epoch, 2});

            summary{epoch, 2} = struct("classification_window_validation", classification_window_validation, ...
                                       "classification_validation", classification_validation, ...
                                       "recognition_validation", recognition_validation);
        end

        if verbose_level > 0
            if mod(epoch, print_step) == 0 || epoch == general_epochs
                if do_validation
                    fprintf("->Epoch %d | Train accuracy: [%.4f, %.4f, %.4f], Validation accuracy: [%.4f, %.4f, %.4f]\n", epoch, ...
                        classification_window_train.accuracy, classification_train.accuracy, recognition_train.accuracy, ...
                        classification_window_validation.accuracy, classification_validation.accuracy, recognition_validation.accuracy);
                else
                    fprintf("->Epoch %d | Train accuracy: [%.4f, %.4f, %.4f]\n", epoch, ...
                        classification_window_train.accuracy, classification_train.accuracy, recognition_train.accuracy);
                end
            end
        end
        
    end
    t_end = toc(t_begin);
end