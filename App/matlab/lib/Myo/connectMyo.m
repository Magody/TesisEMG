function [myoObject, isConnectedMyo] = connectMyo(type)
    %{
        type: string -> "real" or "fake"
    %}

    isConnectedMyo = 1;
    
    
    try
        if type == "real"
            % Nueva conexión
            myoObject = MyoMex();
            %         beep
            % myoObject.myoData.startStreaming();
        elseif type == "fake"
            myoObject = FakeMyoMex();
            % isConnectedMyo = myoObject.myoData.isStreaming;
        end
    catch
        % No conexión posible
        isConnectedMyo = 0;
    end
end