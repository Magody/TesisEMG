classdef Experiment < handle
    
    methods (Static)
        
        function [history_experiments, index_history_experiment] = getTableResults(history_episodes_train, ...
                history_episodes_test, user_id, experiment_id, verbose_level)
            % user stats
            train_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_window_correct'), history_episodes_train('history_classification_window_incorrect'));
            train_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_class_correct'), history_episodes_train('history_classification_class_incorrect'));
            train_metrics_classification_recognition = getMetricsFromCorrectIncorrect(history_episodes_train('history_classification_recognition_correct'), history_episodes_train('history_classification_recognition_incorrect'));


            if verbose_level > 0
                fprintf("Train accuracy, [window=%.4f, classif=%.4f, recog=%.4f]\n", train_metrics_classification_window('accuracy'), train_metrics_classification_class('accuracy'), train_metrics_classification_recognition('accuracy'));
            end

            
            test_metrics_classification_window = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_window_correct'), history_episodes_test('history_classification_window_incorrect'));
            test_metrics_classification_class = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_class_correct'), history_episodes_test('history_classification_class_incorrect'));
            test_metrics_classification_recognition = getMetricsFromCorrectIncorrect(history_episodes_test('history_classification_recognition_correct'), history_episodes_test('history_classification_recognition_incorrect'));

            test_mean_reward = mean(history_episodes_test("history_rewards"));
            if verbose_level > 0
                fprintf("Test accuracy, [window=%.4f, classif=%.4f, recog=%.4f]\n", test_metrics_classification_window('accuracy'), test_metrics_classification_class('accuracy'), test_metrics_classification_recognition('accuracy'));
                fprintf("Test: Mean collected reward %.4f\n", test_mean_reward);
            end

            history_experiments = cell([1, 8]);
            index_history_experiment = 1;

            history_experiments{1, index_history_experiment} = user_id;
            index_history_experiment = index_history_experiment + 1;
            
            history_experiments{1, index_history_experiment} = experiment_id;
            index_history_experiment = index_history_experiment + 1;

            % Train: accuracy window
            history_experiments{1, index_history_experiment} = round(train_metrics_classification_window('accuracy'), 2);
            index_history_experiment = index_history_experiment + 1;

            % Train: accuracy class
            history_experiments{1, index_history_experiment} = round(train_metrics_classification_class('accuracy'), 2);
            index_history_experiment = index_history_experiment + 1;

            % Train: accuracy recognition
            history_experiments{1, index_history_experiment} = round(train_metrics_classification_recognition('accuracy'), 2);
            index_history_experiment = index_history_experiment + 1;
            
            % Test: accuracy window
            history_experiments{1, index_history_experiment} = round(test_metrics_classification_window('accuracy'), 2);
            index_history_experiment = index_history_experiment + 1;

            % Test: accuracy class
            history_experiments{1, index_history_experiment} = round(test_metrics_classification_class('accuracy'), 2);
            index_history_experiment = index_history_experiment + 1;

            % Test: accuracy recognition
            history_experiments{1, index_history_experiment} = round(test_metrics_classification_recognition('accuracy') * 100, 2);
            index_history_experiment = index_history_experiment + 1;

            % Test: test_mean_reward
            history_experiments{1, index_history_experiment} = test_mean_reward;
            index_history_experiment = index_history_experiment + 1;
            
            
            
        end
        
        function plotAndSave(history_episodes_train, saveas_name)
            
            figure(1);
            subplot(1,2,1)
            history_rewards = history_episodes_train('history_rewards')';
            plot(history_rewards(:));
            title("Train: Reward");

            subplot(1,2,2)
            linear_update_costs = [];
            update_costs_by_episode = history_episodes_train('history_update_costs');

            for index_gesture=1:length(update_costs_by_episode)
                costs = update_costs_by_episode{index_gesture};
                linear_update_costs = [linear_update_costs; costs(:)];
            end

            plot(linear_update_costs);
            title("Cost");

            saveas(gcf, saveas_name);
            
            close all;
            
        end
        
    end
    
end