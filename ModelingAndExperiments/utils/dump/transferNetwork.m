function transferNetwork(app, model_reference, classes, userData)

    global qnn_online context_online 
    global params
    global known_dataset_training known_dataset_test

    global stored_qnn stored_extra_classes

    classes_num_to_name = deepCopyMap(model_reference.classes_num_to_name, 'double', 'char');



    classes_to_expand = length(classes);
    classes_actual = prod(model_reference.model.sequential_network.shape_output);


    for i=1:classes_to_expand
        classes_num_to_name(classes_actual + i) = char(classes(i));
    end

    keys_classes = keys(classes_num_to_name);
    output = "";
    for i=1:length(keys_classes)
        key = keys_classes{i};
        output = output +  sprintf("%s->%d\n", classes_num_to_name(key), key);
    end
    app.TextAreaSummary.Value = output;

    if stored_qnn.check > 0
        m_len =  min(length(stored_extra_classes),length(classes));
        exist = true;
        for i=1:m_len
            if classes(i) ~= stored_extra_classes(i)
                exist = false;
                break;
            end
        end
        if exist
            qnn_online = stored_qnn.qnn_online;
            context_online = stored_qnn.context_online;
            known_dataset_training = stored_qnn.known_dataset_training;
            known_dataset_test = stored_qnn.known_dataset_test;

            return;
        end
    end









    output_neurons_new = classes_actual + classes_to_expand;

    sequential_network_length = length(model_reference.model.sequential_network.network);
    network_new = cell([1, sequential_network_length]);

    temp = model_reference.model.sequential_network.network(:);
    for i=1:sequential_network_length-1
        network_new{1, i} = temp{i};

        if mod(i,2) ~= 0

            network_new{1, i}.vdw = 0;
            network_new{1, i}.vdb = 0;
            network_new{1, i}.sdw = 0;
            network_new{1, i}.sdb = 0;
        end
    end
    clear temp;

    % last layer should be new due to different neurons
    last_original = model_reference.model.sequential_network.network{sequential_network_length};
    last_layer = Dense(output_neurons_new, "xavier", prod(network_new{sequential_network_length-1}.shape_output));


    %{
    empty_part_weight = zeros([classes_to_expand, size(last_original.vdw, 2)]);
    empty_part_bias = zeros([classes_to_expand, size(last_original.vdb, 2)]);

    last_layer.vdw = [last_original.vdw; empty_part_weight];
    last_layer.vdb = [last_original.vdb; empty_part_bias];
    last_layer.sdw = [last_original.sdw; empty_part_weight];
    last_layer.sdb = [last_original.sdb; empty_part_bias];

    last_layer.vdw = 0;
    last_layer.vdb = 0;
    last_layer.sdw = 0;
    last_layer.sdb = 0;


    last_layer.weights(1:classes_actual, :) = last_original.weights(1:classes_actual, :);
    last_layer.bias(1:classes_actual, :) = last_original.bias(1:classes_actual, :);

    %}

    last_layer.vdw = 0;
    last_layer.vdb = 0;
    last_layer.sdw = 0;
    last_layer.sdb = 0;

    network_new(sequential_network_length) = {last_layer};

    sequential_new = Sequential({});
    sequential_new.network = network_new;
    sequential_new.shape_input = sequential_new.network{1}.shape_input;
    sequential_new.shape_output = sequential_new.network{sequential_network_length}.shape_output;



    % Sequential(network_new)
    qnn_online = QNeuralNetwork(Sequential({}), sequential_new, ...
        model_reference.model.nnConfig, ...
        model_reference.model.qLearningConfig, ...
        model_reference.model.functionExecuteEpisode);

    qnn_online.transferGameReplay(model_reference.model.gameReplay);
    qnn_online.setCustomRunEpisodes(@customRunEpisodesEMG);            


    global labels;

    labels = cell([1, length(classes_num_to_name)]);
    for i=1:length(classes_num_to_name)
        labels{i} = classes_num_to_name(i);
    end

    context_online = generateContext(params, classes_num_to_name);

    dataset_complete = userData.training;
    dataset_complete_test = userData.testing;
    known_dataset_training = {};
    known_dataset_test = {};
    known_classes = string(values(model_reference.classes_num_to_name));
    known_classes_test = string(values(classes_num_to_name));

    for index_dataset_complete=1:length(dataset_complete)
        gesture = dataset_complete{index_dataset_complete}.gestureName;
        gesture_test = dataset_complete_test{index_dataset_complete}.gestureName;
        
        if searchStringInArray(known_classes, gesture) ~= -1
            known_dataset_training = [known_dataset_training, dataset_complete(index_dataset_complete)];
        end
        if searchStringInArray(known_classes_test, gesture_test) ~= -1
            known_dataset_test = [known_dataset_test, dataset_complete_test(index_dataset_complete)];
        else
            fprintf("%s\n", string(gesture_test));
        end
    end






    stored_qnn.qnn_online = qnn_online;
    stored_qnn.context_online = context_online;
    stored_qnn.known_dataset_training = known_dataset_training;
    stored_qnn.known_dataset_test = known_dataset_test;
    stored_qnn.check = 1;




end