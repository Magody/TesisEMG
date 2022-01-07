function myTimer = timerWaitbar(timeGesture)

myTimer = timer('Name', 'waitbarTimer');
myTimer.ExecutionMode = 'fixedRate'; %fixedSpacing
% myTimer.ExecutionMode = 'fixedRate';
myTimer.Period = 0.1;

%
numExecutions = floor(timeGesture / 0.1); % depende del tiempo del gesto
myTimer.TasksToExecute = numExecutions;

myTimer.TimerFcn = @(timer, evnt)disp(timer.TasksExecuted);

end
