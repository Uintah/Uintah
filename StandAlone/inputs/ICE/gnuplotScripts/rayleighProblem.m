%_________________________________
% This matLab file plots the velocity profile
% of the rayleigh problem with and without dimensions
% Reference:  Incompressible Flow, by Panton pg 177
%
%  - first run sus -ice rayleigh.ups
%  - cd rayleigh.uda
%  - matlab ../inputs/ICE/gnuplotScrips/rayleigh.m
%_________________________________
%  HardWired Variables  ( Tweak these)
clear all;
close all;
timesteps={405,805, 1206,1607,2008,2409,2810,3211,3612};
time = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
viscosity = 0.01;
vel_CC_initial = 10.0;

%_________________________________
% Loop over all the timesteps
for( t = 1:length(timesteps) )
  
  %Load the data into the arrays  
  here = timesteps{t};
  path = sprintf('BOT_Advection_after_BC/%d/patch_combined/Mat_0/X_vel_CC',here);
  load(path);
  data =importdata(path);
  y    =data(:,1);
  uvel =data(:,2);
  
  % computes quantities
  vel_ratio = uvel/vel_CC_initial;
  eta = y/sqrt(4.0 * viscosity * time{t});
  vel_ratio_exact =( 1.0 - erf( y/(2.0 * sqrt(viscosity * time{t})) ) );
  
  %______________________________
  % Plot the results from each timestep
  % onto 2 plots
  subplot(2,1,1),plot(uvel, y)
  xlabel('uvel')
  ylabel('y')
  legend('computed');
  title('Rayleigh Problem');
  grid on;
  hold on;
  
  subplot(2,1,2),plot(vel_ratio_exact,eta);
  hold on;
  plot(vel_ratio,eta,'r:+')
  xlabel('u/U0'); 
  ylabel('eta');
  legend('exact', 'computed');
  grid on;
  axis([0 1 0 3]);
  
  pause
  clear y, uvel, vel_ratio, eta, vel_ratio_exact;
end
hold off;

