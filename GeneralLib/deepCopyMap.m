function map_copy = deepCopyMap(map_original, type_key, type_value)

    map_copy = containers.Map('KeyType',type_key,'ValueType',type_value);
    list_keys = keys(map_original);
    for i=1:length(list_keys)
        key = list_keys{i};
        value = map_original(key);
        map_copy(key) = value;
    end
end

