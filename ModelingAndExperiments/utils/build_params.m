function [params, hyperparams] = build_params(params_experiment_row, mode, verbose_level)

	index_begin = 2;
	
	params = struct();

	params.porc_training = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.porc_validation = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.list_users = table2matrix(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
    params.window_size = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	params.stride = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
	hyperparams.seed_rng = table2array(params_experiment_row(1, index_begin));
    generate_rng(hyperparams.seed_rng);
	index_begin = index_begin + 1;
    
    hyperparams.gameReplayStrategy = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
    hyperparams.epsilon = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
    rewards = table2matrix(params_experiment_row(1, index_begin));
	hyperparams.rewards = struct('correct', rewards(1), 'incorrect', rewards(2));  % struct('correct', 1, 'incorrect', -1);
	index_begin = index_begin + 1;
    
    hyperparams.decay_rate_alpha = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	hyperparams.loss_type = table2string(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	hyperparams.experience_replay_reserved_space = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
    hyperparams.batch_size = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
	hyperparams.general_epochs = table2array(params_experiment_row(1, index_begin)); % epochs inside each NN
	index_begin = index_begin + 1;
    
    hyperparams.inner_epochs = table2array(params_experiment_row(1, index_begin)); % epochs inside each NN
	index_begin = index_begin + 1;
    
	hyperparams.interval_for_learning = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
	
	hyperparams.learning_rate = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
	hyperparams.gamma = table2array(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
	% C-[1, 3]-8-0-1 | A-relu | P-max-[1,2]
	layers_conv_network = split(table2string(params_experiment_row(1, index_begin)), "|");
	index_begin = index_begin + 1;
	num_layers_conv_network = length(layers_conv_network);
	
	input_dense = -1;
    hyperparams.sequential_conv_network = Sequential({});
    
	if num_layers_conv_network > 1
	
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
        hyperparams.sequential_conv_network = Sequential(conv_network);
		input_dense = prod(hyperparams.sequential_conv_network.shape_output);% if convolutional network exist, sequential_conv_network.shape_output;
		
	end
	
	layers_network = split(table2string(params_experiment_row(1, index_begin)), "|");
	num_layers_network = length(layers_network);
	network = cell([1, num_layers_network]);
	index_begin = index_begin + 1;
	
	
	% D-64-kaiming-input_dense | A-relu | Dropout-0.3 | D-64-kaiming- | A-relu | D-6-xavier
	
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
        elseif layer_type == "Dropout"    
            drop = str2num(layer_config(2));
            network{l} =  Dropout(drop);
	    end
	end
	
	hyperparams.sequential_network = Sequential(network);
    
    
	params.model_name = table2string(params_experiment_row(1, index_begin));
	index_begin = index_begin + 1;
    
    
    params.verbose_level = verbose_level;
    params.debug_steps = 10;
    
end

