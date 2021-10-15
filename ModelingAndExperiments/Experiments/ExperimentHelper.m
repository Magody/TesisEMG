classdef ExperimentHelper < handle
    
    methods (Static)
        
        function saveModel(model_name, model, history, summary, context)

            % save model
            classes_num_to_name = context('classes_num_to_name');
            classes_name_to_num = context('classes_name_to_num');

            save(model_name, "model", "history", "summary", ...
                             "classes_num_to_name", ...
                             "classes_name_to_num");
        end
        
        function [summary, responses] = testModelIndividual(model, ...
                run_as_validation, context, verbose_level)
            
            summary.classification_window_test = 0;
            summary.classification_test = 0;
            summary.recognition_test = 0;
            
            type_execution = 3;
            if run_as_validation
                type_execution = 2;
            end

            history_test = model.runEpisodes(@getRewardEMG, type_execution, context, verbose_level-1);

            responses = history_test.history_responses;

            if run_as_validation
                [classification_window_test, classification_test, recognition_test] = Experiment.getEpisodesEMGMetrics(history_test);
                summary.classification_window_test = classification_window_test.accuracy;
                summary.classification_test = classification_test.accuracy;
                summary.recognition_test = recognition_test.accuracy;
                if classification_test.accuracy < 0.92
                    fprintf("%s->Test accuracy: [%.4f, %.4f, %.4f]\n", user_folder, ...
                        classification_window_test.accuracy, classification_test.accuracy, recognition_test.accuracy);
                end
            end
        end
        
    end
    
end