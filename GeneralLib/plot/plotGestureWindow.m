function plotGestureWindow(ID, emg, point_start, window_size)

figure(ID);
clf;
plot(emg)
xlim([0 length(emg)])
ylim([-1.2 1.2])
grid on

% rectangle
rectangle('Position',[point_start -1  window_size 2],'EdgeColor','r')


end

