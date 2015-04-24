%_________________________________
% 05/01/07   --Todd
% This matLab script plots scaling curves
% You must run Uintah/scripts/extractScalingData first

close all;
clear all;

%________________________________
% USER INPUTS
%'
datadir='Data';
mult=8;

normFont  = 14;
bigFont   = normFont+2;
smallFont = normFont-2;

dirname{1} = sprintf('small/output/scalingData');
dirname{2} = sprintf('medium/output/scalingData');
dirname{3} = sprintf('large/output/scalingData');

legendText = {'Strong' 'Weak'};

%________________________________
%
set(0,'DefaultFigurePosition',[0,0,1024,768]);
dsort=[];
psizea=[];

for i=1:length(dirname)
   d = importdata(dirname{i},' ', 1);
   data = d.data;
  
  %compute relative problem size
  psize=mult^i./data(:,1);
  if i==1
    psizea=psize;
  end
  
  %compute (R)un index for these data
  Rindex=ones(length(psize),1).*i;
  
  %append problem size and Rindex to the sorted data
  dsort=[dsort; Rindex, psize, data ];
end

dsort
% number of cores and mean time per timestep
cores = dsort(:,3);
meanTime = dsort(:,6);
  
psizea
%__________________________________
% strong scaling
disp('__________________________________ Strong Scaling' )
for i=1:length(dirname)
  dirname{i};
  Rindex = find( dsort(:,1) == i);  % run index 
  
  % cores and the mean time per timestep
  cores(Rindex)
  meanTime(Rindex)
 
  % plot #core vs mean time per timestep
  loglog( cores(Rindex), meanTime(Rindex), '-ok','LineWidth',2);
  %pause
  
  hold on;
  
  %weak placehnewer in legend
  loglog(0,0,'--ok','LineWidth',1);
end

%__________________________________
% weak scaling
disp('__________________________________ Weak Scaling' )
for i=1:length(psizea)
  psizea(i);
  Rindex = find(dsort(:,2)>psizea(i)-.0001 & dsort(:,2)>psizea(i)<psizea(i)+.0001)
  
  % cores and the mean time per timestep
  cores(Rindex)
  meanTime(Rindex)
  
  loglog( cores(Rindex), meanTime(Rindex),'--ok','LineWidth',1);
end

%__________________________________
% set plot variables on;

ylabel('Mean Time Per Timestep (s)','fontsize', normFont, 'FontName','Times')
xlabel('Cores','fontsize', normFont, 'FontName', 'Times')
title('Uintah:MiniAero - Riemann 3D : Titan','fontsize', bigFont, 'FontName', 'Times')

str1(1) = {'Runge Kutta: 4'};
str1(2) = {'Viscous Terms Included'};
str1(3) = {'Unified Scheduler'};
text(64,2,str1, 'FontSize', normFont, 'FontName', 'Times')
set(gca,'FontSize',normFont, 'FontName', 'Times');

legend(legendText,'fontsize',normFont,'Location','northeast');

xlim([32 131072 ]);
ylim([0.8 50]);

set(gca,'XTick',[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 786432]);
set(gca,'XTickLabel',{ '16', '32', '64', '128', '256', '512', '1K', '2K', '4K', '8K', '16K', '32.7K', '65.5K', '131K', '262.1K', '524.2K', '786.4K'});

disp( 'Press return to generate a hardcopy' )
pause;
print('-depsc','miniAeroScalingUnified_RK4_spider.eps');
print('-dpng','miniAeroScalingUnified_RK4_spider.png');



