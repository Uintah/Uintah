%_________________________________
% 05/01/07   --Todd
% This matLab script plots scaling curves
% You must run Uintah/scripts/extractScalingData first

close all;
clear all;

%________________________________
% USER INPUTS
%'
dir1 = 'study0/data'
dir2 = 'study1/data'
dir3 = 'study4/data'
dir4 = 'autoPatch/data'
dir5 = 'fixedPatch/data'

legendText = {'Patch layout (16,16,16)','Patch layout (16,16,16) Load Bal.:DLB','Patch layout (16,16,16) Load Bal.:DLB, Today',...
              'autoPatch (# patches = # procs), DLB','autoPatch (# patches = # procs), DLB, Today'}

%________________________________
%
d1  = importdata(dir1,' ');
d2  = importdata(dir2,' ');
d3  = importdata(dir3,' ');
d4  = importdata(dir4,' ');
d5  = importdata(dir5,' ');


dsort1 = sortrows(d1.data,1)
dsort2 = sortrows(d2.data,1)
dsort3 = sortrows(d3.data,1)
dsort4 = sortrows(d4.data,1)
dsort5 = sortrows(d5.data,1)

loglog(dsort1(:,1), dsort1(:,2),'-or',dsort3(:,1), dsort3(:,2),'-*g',dsort5(:,1), dsort5(:,2),'-sg',dsort2(:,1), dsort2(:,2),'-xb',dsort4(:,1), dsort4(:,2),'-*b')
grid on;
ylabel('Elapsed Time [sec.]')
xlabel('Processors')
title('MPMICE: Expode\_IS.ups, Single Level, Fixed size (200^3)')
legend(legendText);