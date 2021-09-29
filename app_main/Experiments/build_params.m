function [params, nnConfig, qLearningConfig] = build_params(params_experiment_row, context)

	index_begin = 2;
	
	params = struct();

	params.RepTraining = table2array(params_experiment_row(1, index_begin));
	assignin('base','RepTraining',  params.RepTraining); % initial value
	context('RepTraining') = params.RepTraining;
	index_begin = index_begin + 1;
	
	params.RepValidation = table2array(params_experiment_row(1, index_begin));
	context('RepValidation') = params.RepValidation;
	index_begin = index_begin + 1;
	
	params.RepTesting = table2array(params_experiment_row(1, index_begin));
	context('RepTesting') = params.RepTesting;
	index_begin = index_begin + 1;
	
	params.rangeDownTrain = table2array(params_experiment_row(1, index_begin));
	context('rangeDownTrain') = params.rangeDownTrain;
	index_begin = index_begin + 1;
	
	params.rangeDownValidation = table2array(params_experiment_row(1, index_begin));
	context('rangeDownValidation') = params.rangeDownValidation;
	index_begin = index_begin + 1;
	
	params.rangeDownTest = table2array(params_experiment_row(1, index_begin));
	context('rangeDownTest') = params.rangeDownTest;
	index_begin = index_begin + 1;
	
	params.list_users = str2num(string(table2cell(params_experiment_row(1, index_begin)))); % [8 200]; 1:306;
	context('list_users') = params.list_users;
	params.num_users = length(params.list_users);
	context('num_users') = params.num_users;
	index_begin = index_begin + 1;
	
	params.list_users_validation = str2num(string(table2cell(params_experiment_row(1, index_begin)))); % [1 2]; 1:306;
	context('list_users_validation') = params.list_users_validation;
	params.num_users_validation = length(params.list_users_validation);
	context('num_users_validation') = params.num_users_validation;
	index_begin = index_begin + 1;
	
	params.list_users_test = str2num(string(table2cell(params_experiment_row(1, index_begin)))); % [1 2]; 1:306;
	context('list_users_test') = params.list_users_test;
	params.num_users_test = length(params.list_users_test);
	context('num_users_test') = params.num_users_test;
	index_begin = index_begin + 1;
	
	
	params.seed_rng = table2array(params_experiment_row(1, index_begin));
    generate_rng(params.seed_rng);
	index_begin = index_begin + 1;
	
	params.decay_rate_alpha = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.epochs = table2array(params_experiment_row(1, index_begin)); % epochs inside each NN
	index_begin = index_begin + 1;
	
    rewards = str2num(string(table2cell(params_experiment_row(1, index_begin))));
	params.rewards = struct('correct', rewards(1), 'incorrect', rewards(2));  % struct('correct', 1, 'incorrect', -1);
	context('rewards') = params.rewards;
	index_begin = index_begin + 1;
	
	params.gameReplayStrategy = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.experience_replay_reserved_space = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.epsilon_initial = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.loss_type = string(table2cell(params_experiment_row(1, index_begin)));
	index_begin = index_begin + 1;
	
	params.window_size = table2array(params_experiment_row(1, index_begin));
	context('window_size') = params.window_size;
	assignin('base','WindowsSize',  params.window_size);
	index_begin = index_begin + 1;
	
	params.stride = table2array(params_experiment_row(1, index_begin));
	context('stride') = params.stride;
	assignin('base','Stride',  params.stride);
	index_begin = index_begin + 1;
	
	params.learning_rate = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.batch_size = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.gamma = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.interval_for_learning = table2array(params_experiment_row(1, index_begin));
	context('interval_for_learning') = params.interval_for_learning;
	index_begin = index_begin + 1;



	total_episodes = params.RepTraining * params.num_users;
	total_episodes_test = params.RepTesting * params.num_users_test;
	
	
	nnConfig = NNConfig(params.epochs, params.learning_rate, params.batch_size, params.loss_type);
	nnConfig.decay_rate_alpha = params.decay_rate_alpha;


	
	% C-[1, 3]-8-0-1 | A-relu | P-max-[1,2]
	
	layers_conv_network = split(string(table2cell(params_experiment_row(1, index_begin))), "|");
	index_begin = index_begin + 1;
	num_layers_conv_network = length(layers_conv_network);
	
	conv_network = {};
	input_dense = -1;
    params.sequential_conv_network = Sequential({});
    
	if num_layers_conv_network > 0
	
		conv_network = cell([1, num_layers_conv_network+1]);
		conv_network{num_layers_conv_network+1} = Reshape();
		
		
        shape_input = [1, params.window_size, 8];
		
		for l=1:num_layers_conv_network
		    layer_definition = layers_conv_network(l);
		    layer_config = split(layer_definition, "-");
		    layer_type = layer_config(1);
		    num_configs = length(layer_config);
		    
		    if layer_type == "C"
                shape_convolution = str2num(layer_config(2)); %#ok<*ST2NM>
                filters = str2num(layer_config(3));
                padding = 0;
                stride = 1;
                if l == 1
                    conv_network{l} = Convolutional(shape_convolution, filters, padding, stride, shape_input);
                else
                    conv_network{l} = Convolutional(shape_convolution, filters, padding, stride);
                end
			
		    elseif layer_type == "A"
                activation_function = string(layer_config(2));
                if activation_function == "softmax"
                    conv_network{l} =  ActivationOnlyForward(activation_function);
                else
                    conv_network{l} =  Activation(activation_function);
                end

            elseif layer_type == "P"
                pooling_type = string(layer_config(2));

                if num_configs > 2
                    shape_pooling = str2num(layer_config(3));
                    conv_network{l} =  Pooling(pooling_type, shape_pooling);
                else
                    conv_network{l} =  Pooling(pooling_type);
                end
		    end
        end
        params.sequential_conv_network = Sequential(conv_network);
		input_dense = prod(params.sequential_conv_network.shape_output);% if convolutional network exist, sequential_conv_network.shape_output;
		
	end
	
	layers_network = split(string(table2cell(params_experiment_row(1, index_begin))), "|");
	num_layers_network = length(layers_network);
	network = cell([1, num_layers_network]);
	
	
	% D-64-kaiming-input_dense | A-relu | D-64-kaiming- | A-relu | D-6-xavier
	
	for l=1:num_layers_network
	    layer_definition = layers_network(l);
	    layer_config = split(layer_definition, "-");
	    layer_type = layer_config(1);
	    num_configs = length(layer_config);
	    if layer_type == "D"
	        output = str2num(layer_config(2));
	        
	        init_weights_algorithm = "xavier";
            if num_configs >= 2
                init_weights_algorithm = string(layer_config(3));
            end
            if num_configs == 2 || num_configs == 3
                network{l} = Dense(output, init_weights_algorithm);
            end
	        
            if num_configs > 3
                config_value = layer_config(4);
                
                if config_value == "auto"
                    network{l} = Dense(output, init_weights_algorithm, input_dense);
                else
                    input_dense_custom = str2num(config_value);
                    if isempty(input_dense_custom)
                        network{l} = Dense(output, init_weights_algorithm);
                    else
                        network{l} = Dense(output, init_weights_algorithm, input_dense_custom);
                    end
                end
            end
        
            
	        
	        
	        
	    elseif layer_type == "A"
	        activation_function = string(layer_config(2));
            if activation_function == "softmax"
                network{l} =  ActivationOnlyForward(activation_function);
            else
                network{l} =  Activation(activation_function);
            end
	    end
	end
	
	params.sequential_network = Sequential(network);

	qLearningConfig = QLearningConfig(params.gamma, params.epsilon_initial, params.gameReplayStrategy, params.experience_replay_reserved_space, total_episodes);
	qLearningConfig.total_episodes_test = total_episodes_test;


    
    
end

