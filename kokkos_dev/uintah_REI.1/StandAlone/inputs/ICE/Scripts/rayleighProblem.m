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

uda      = sprintf('rayleigh.uda.001')
viscosity = 0.01;
vel_CC_initial = 10.0;
dy = 0.35/50;

% lineExtract options
options = '-istart 25 0 0 -iend 25 50 0 -m 0'

%________________________________
%  extract the physical time of each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;

%_________________________________
% Loop over all the timesteps
for( t = 1:nDumps )
  %t = input('input timestep')
  time =physicalTime(t)
  %use line extract to pull out the data
  c1 = sprintf('lineextract -v vel_CC -timestep %i %s -o vel_tmp -uda %s',t,options,uda)
  [status1, result1]=unix(c1)
  
  % rip out [] from velocity data
  c2 = sprintf('sed ''s/\\[//g'' vel_tmp | sed ''s/\\]//g'' >vel');
  [status2, result2]=unix(c2);

  % import the data into arrays
  vel1    = importdata('vel'); 
  y = vel1(:,2) * dy;
  uvel = vel1(:,4);    
  vvel = vel1(:,5);
  wvel = vel1(:,6);
 
  % computes quantities
  vel_ratio = uvel/vel_CC_initial;
  eta = y/sqrt(4.0 * viscosity * time);
  vel_ratio_exact =( 1.0 - erf( y/(2.0 * sqrt(viscosity * time)) ) );
  
  %______________________________
  % Plot the results from each timestep
  % onto 2 plots
  subplot(2,1,1),plot(uvel, y)
  xlabel('u velocity')
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
  
  %pause
  clear y, uvel, vel_ratio, eta, vel_ratio_exact;
end
hold off;

