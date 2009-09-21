%_________________________________
% 06/22/04   --Todd
% This matLab script generates 1D plots
% from two different uda directories
%
%  NOTE:  You must have 
%  setenv MATLAB_SHELL "/bin/tcsh -f"
%  in your .cshrc file for this to work
%_________________________________
clear all;
clear functions;
close all;
setenv('LD_LIBRARY_PATH', ['/usr/lib']);

%________________________________
% define the uda dirctories here
uda1 = sprintf('pulse_Lodi2_xdir.uda.000');         %----change this
uda2 = sprintf('pulse_Neumann2_xdir.uda.000');        %----change this
desc     = sprintf('BC comparison');
uda1_txt = sprintf('suspect');
uda2_txt = sprintf('good');

[ans, test]=unix('printenv MATLAB_SHELL');

unix('/bin/rm ts_1 ts_2');
command1 = sprintf('cd %s; ../inputs/ICE/gnuplotScripts/findTimesteps>&../ts_1',uda1);
command2 = sprintf('cd %s; ../inputs/ICE/gnuplotScripts/findTimesteps>&../ts_2',uda2);
[status1, result1]=unix(command1)
[status2, result2]=unix(command2)
unix('sleep 2');

%_________________________________
%  bulletproofing
if(status1 ~= 1 | status2 ~=1 )
    fprintf(' Initial unix commands failed ');
    exit;
end

timesteps1 = importdata('ts_1');   % from uda 1
unix('sleep 2');
timesteps2 = importdata('ts_2');   % from uda 1
unix('sleep 2');

set(0,'DefaultFigurePosition',[0,0,1024,768]);

%_________________________________
% Loop over all the timesteps
%for( t = 1:length(timesteps2) )
for(c = 1: 30 )
  t = input('input timestep')      %----if you only want to look at a few timesteps
  
  %______________________________
  %Load the data into the arrays 
  here = timesteps1(t);
  here2 = timesteps2(t);
  p1 = sprintf('%s/BOT_equilibration/%d/L-0/patch_0/Press_CC_equil',        uda1,here);
  %p1 = sprintf('%s/BOT_explicit_Pressure/%d/L-0/patch_0/Press_CC',         uda1,here);
  t1 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/Temp_CC',    uda1,here);
  r1 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/rho_CC',     uda1,here);
  v1 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/sp_vol_CC',  uda1,here);
  %v1 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/Y_vel_CC',  uda1,here);
  
  p2 = sprintf('%s/BOT_equilibration/%d/L-0/patch_0/Press_CC_equil',       uda2,here2);
  %p2 = sprintf('%s/BOT_explicit_Pressure/%d/L-0/patch_0/Press_CC',        uda2,here2);
  t2 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/Temp_CC',   uda2,here2);
  r2 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/rho_CC',    uda2,here2);
  v2 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/sp_vol_CC', uda2,here2);
  %v2 = sprintf('%s/BOT_Advection_after_BC/%d/L-0/patch_0/Mat_0/Y_vel_CC', uda1,here2);
  
  press1  = sortrows(importdata(p1),1);
  temp1   = importdata(t1);
  rho1    = importdata(r1);
  vel1    = importdata(v1);
  press2  = sortrows(importdata(p2),1);
  temp2   = importdata(t2);
  rho2    = importdata(r2);
  vel2    = importdata(v2);
  %____________________________

  
  
  %____________________________
  %  temperature
  subplot(2,2,1), plot(temp1(:,1),temp1(:,2),'--r',temp2(:,1),temp2(:,2),'--g')
  legend(uda1_txt, uda2_txt);
  %axis([0 1e-3 295,625])
  xlabel('x')
  ylabel('temp')
  grid on;
  
  %______________________________
  % Plot the pressure
  subplot(2,2,2),plot(press1(:,1),press1(:,2),'--r',press2(:,1),press2(:,2),'--g')
  %axis([0 1e-3 0 15])
  xlabel('x')
  ylabel('pressure')
  title(desc);
  grid on;
  
  %_____________________________
  %  Density
  subplot(2,2,3), plot(rho1(:,1),rho1(:,2), '--r', rho2(:,1),rho2(:,2), '--g')
  %xlim([0 1e-3])
  %axis([0 1e-3 1.75 1.9])
  xlabel('x')
  ylabel('rho')
  grid on;
  
  %____________________________
  %   specific volume
  subplot(2,2,4), plot(vel1(:,1),vel1(:,2), '--r', vel2(:,1), vel2(:,2), '--g')
  %xlim([0 1e-3])
  %axis([0 5 -10 10])
  ylabel('sp_vol');
  grid on;
  
  M(t) = getframe(gcf);
  
  clear press1,press2,p1,p2,here,here2, temp1, temp2, t1,t2, rho1, rho2, r1,r2;
  clear v1, v2, vel1, vel2;
  %pause; 
  
end
%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'movie.avi','fps',5,'quality',100);
%clear all
