function map_copy = deepCopyMapInverse(map_original)
    map_copy = containers.Map();
    list_keys = keys(map_original);
    for i=1:length(list_keys)
        key = list_keys{i};
        value = map_original(key);
        map_copy(value) = key;
    end
end

