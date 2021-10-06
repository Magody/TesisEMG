clc;
clear all;
close all;

%% lib

path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework"; % "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework";% "C:\Users\Magody\Documents\GitHub\MATLABMagodyFramework\magody_framework"; "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
addpath(genpath(path_to_framework));
addpath(genpath('utils'));
addpath(genpath('RLSetup'));

%% parameters GA
verbose_level = 2;

generations = 5;  % its just a stop step in case the program cant find solution
max_population = 10;  % while more high, more posibilities to find a good combination faster, but more processing
num_parents_to_select = 2;  % is better tu select very little, no more than 10% of max population
mutation_rate = 0.9;

amount_examples = 100;

% examples of parameters
gens_set = [linspace(1, 20, amount_examples); ... 
              linspace(1, 10, amount_examples); ...  
              linspace(0.0003, 0.01, amount_examples); ...  
              linspace(0, 1, amount_examples); ...  
              linspace(6, 64, amount_examples); ... 
              linspace(6, 64, amount_examples); ...  
              linspace(0, 0.9, amount_examples); ... 
              linspace(0, 0.9, amount_examples); ... 
              linspace(16, 256, amount_examples); ... 
              linspace(0.001, 0.3, amount_examples); ... 
              linspace(20, 1000, amount_examples)
              ];                
                
          
% num of parameters to optimice (alpha, neurons_hidden1)
num_gens = length(gens_set(:,1));

population = Population(max_population, gens_set, mutation_rate);
population.generate_initial_population(num_gens);


%% Populate
history_fitness_mean = [];
history_change = [];
register_change = cell(1, num_gens);  % each row is independent

t_start = tic;
for generation=1:generations
    
    if verbose_level >= 1
       fprintf("\n\n*********************\n");
       fprintf("Generation %d of %d\n", generation, generations);
       fprintf("*********************\n"); 
    end
    
    fitness = population.fitness_function(verbose_level-1);
    
    % being elitist means that the parents are the best 2 (maybe a little more)
    
    best_individuals_index = Population.select_best_individuals_by_elitist(fitness, num_parents_to_select);
    parents_selected = population.chromosomes(best_individuals_index);
    
    %%%%%%%%%%%%%%RESULTS ACTUAL GENERATION%%%%%%%%%%%%%%%%%
    if verbose_level >= 1
        fprintf("Best result for generation %d\n", generation);
        % disp("Individuals (chromosomes)");
        % disp(parents);
        disp("fitness for individuals");
        disp(fitness(best_individuals_index));
    end
    history_fitness_mean(generation) = mean(fitness);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    if Population.hasReachedTheTop(fitness(best_individuals_index))
       disp("FUNCTION HAS CONVERGED");
       break;
    end
    
    if generation < generations
        offspring_crossover = Population.crossover(parents_selected, max_population-num_parents_to_select);
    
        % the change for the alele of gen, higher at the begining (exploration), 
        % lower at the end (lower exploration, high explotation)
        reductor_factor = log(3+generation/3);  % change = R/reductor_factor decreases in the next generations

        [offspring_mutated, register_change] = population.mutate(offspring_crossover, reductor_factor, register_change);  % Creating the new population based on the parents and offspring.


        population.chromosomes(1:num_parents_to_select) = parents_selected;
        population.chromosomes(num_parents_to_select+1:end) = offspring_mutated;
    end
end
elapsed_time = toc(t_start)/60;

fprintf("\nTime: %3.3f [minutes]\n", elapsed_time);

final_generations = numel(history_fitness_mean);


% % Best results
disp("Answer");
index_best_chromosome = Population.select_best_individuals_by_elitist(fitness, 1);
best_chromosome = population.chromosomes(index_best_chromosome);
disp(best_chromosome);
fprintf("Fitness %f\n", fitness(index_best_chromosome));


model_dir = "Experiments/gens.mat";
save(model_dir,'best_chromosome');

%% figures
figure(1);
plot(1:final_generations, history_fitness_mean)

figure(2);
for i=1:num_gens
    subplot(1, num_gens, i);
    plot(1:length(register_change{i}), register_change{i});
end

grid on;


