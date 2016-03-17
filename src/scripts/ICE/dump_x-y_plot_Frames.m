%_________________________________
% 05/01/07   --Todd
% This matLab script makes a movie of the mean time per timestep.

close all;
clear all;

%________________________________
% load pre parsed data into arrays
dump_timestep  = importdata('outputTimesteps');
data = importdata('data');
figure
set(gcf,'Position',[0,0,700,510])

for i = 1:length(dump_timestep)

  data_indx = max(find(data(:,1) < dump_timestep(i)) )
  physicalTime = data(data_indx,2)
  
  data_cropped = data(1:data_indx,1:3);
  plot(data_cropped(:,2), data_cropped(:,3),'-r')
  xlim([0 2.5e-3]);
  ylim([0 2.5]);
  grid on
  ylabel('Mean Time per Timestep [sec.]','FontSize',14,'FontWeight','bold')
  xlabel('Physicial Time [sec.]','FontSize',14,'FontWeight','bold')
  
  t = sprintf('%4.3d msec',physicalTime*1000)
  title(t)
  f_name = sprintf('movie.%04d.ppm',i-1)
  F = getframe(gcf);
  [X,map] = frame2im(F);
  imwrite(X,f_name)
  %pause
end
