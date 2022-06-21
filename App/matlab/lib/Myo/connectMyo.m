function [myoObject, isConnectedMyo] = connectMyo(type)
    %{
        type: string -> "real" or "fake"
    %}

    isConnectedMyo = 1;
    disp("Connecting myo...SIM:" + type);
    
    
    try
        if type == "real"
            % Nueva conexión
            myoObject = MyoMex();
            %         beep
            % myoObject.myoData.startStreaming();
            isConnectedMyo = 1;
             disp("Connected real");
        elseif type == "fake"
            myoObject = FakeMyoMex();
            isConnectedMyo = myoObject.myoData.isStreaming;
             disp("Connected fake");
        end
    catch ME
        % No conexión posible
        disp("Error connecting");
        disp(ME);
        isConnectedMyo = 0;
    end
end