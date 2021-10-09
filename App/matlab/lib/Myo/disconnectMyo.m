function disconnectMyo(myoObject)
% Function to stop myo for streaming data, is required to delete object
% twice just to be sure.

try
    % Stopping myoObject
    myoObject.myoData.stopStreaming();
    myoObject.delete;
    clear myoObject
    
    fprintf('Connection with myo finished!\n');
catch
    try
        myoObject.myoData.stopStreaming();
        myoObject.delete;
        clear myoObject
    catch
    end
end

