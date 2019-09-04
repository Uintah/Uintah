#! /usr/bin/octave -qf

%______________________________________________________________________
% pistonVal_Solution.m
% This script generates the exact solution to the
% piston validation problem. 
%______________________________________________________________________

clear all;
close all;

%_____________________________________________________________________
% Problem specific variables

problemUnits = "mks";   % options:  cgs, mks    <<<<<<<<<<< choose one

if(problemUnits == "cgs")
  uda             = sprintf('imp_pistonVal_cgs.uda');    % uda to compare against
  piston_vel      = 0.001;
  piston_height   = 0.001;
  resolution      = 200;
  delta_y         = 0.0025./resolution;      % domain length/# cells
  piston_stopTime = 0.8;                     % when the piston stops
  p_initial       = 1.01325;                 % initial pressure in chamber
  delT_dump       = 1e-2;                    % how often data is dumped
  desc = 'Piston Validation (cgs units) problem, Pressure probe at the bottom of the cylinder';
elseif (problemUnits == "mks")
  uda             = sprintf('imp_pistonVal_mks.uda');    % uda to compare against
  piston_vel      = 10.0;
  piston_height   = 0.1;
  resolution      = 200;
  piston_stopTime = 0.008
  delta_y         = 2.5e-1./resolution;      % domain length/# cells      
  p_initial       = 101325;                  % initial pressure in chamber 
  delT_dump       = 1e-4;                    % how often data is dumped 
  desc = 'Piston Validation (mks units) problem, Pressure probe at the bottom of the cylinder';  
end

%__________________________________
h_initial1     = piston_height - delta_y;   % initial height of the piston.
h_initial2     = piston_height;
gamma          = 1.4;   

%________________________________
% do the Uintah utilities exist?
[s0, r0]=system(" which puda");
[s1, r1]=system(" which lineextract");
[s2, r2]=system(" which timeextract");

if( s0 != 0 || s1 != 0 || s2 != 0 )
  disp('Cannot execute uintah utilites puda or lineextract or timeextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract/timeextract) have been compiled');
  return;
end

%______________________________________________________________________

%________________________________
%  extract the physical time of each output timestep
c0 = sprintf('puda -timesteps %s | grep ''^[0-9]'' | awk ''{print $2}'' > tmp 2>&1 ',uda);
[status0, result0] = unix(c0);
physicalTime  = load('tmp');


%_____________________________________________________________________
% exact solution
% gamma law for the gas.
% P_chamber = p_initial(Vol_initial/Vol_t) ^ gamma
% Since we don't know where the exact surface is we compute
% an upper and lower bound.

time_sec = physicalTime;

distance = (piston_vel * time_sec);
% for time> piston_stopTime
distance( time_sec>=piston_stopTime ) = (piston_vel * piston_stopTime);


tmp = (h_initial1./( h_initial1 - distance));
p_chamber_1 = p_initial * tmp.^gamma;

tmp = (h_initial2./( h_initial2 - distance));
p_chamber_2 = p_initial * tmp.^gamma;


% Write out the exact solution
file_id=fopen('pistonVal.dat','w');
fprintf(file_id,'%6.4f %15.3f  %15.3f\n',[time_sec;p_chamber_1;p_chamber_2]);
fclose(file_id);

%__________________________________
% extract the pressure at cell 1

c2 = sprintf( 'timeextract --material 0 --variable press_CC --index 0 2 0 --out press -uda %s', uda);
[status, result] = unix(c2);
pressArray = load('press');
time_sim  = pressArray(:,1); 
press_sim = pressArray(:,2);

  
%________________________________
% plot up the the exact solution and 
% simulation results
figure
plot(time_sec, p_chamber_1, time_sec, p_chamber_2, time_sim, press_sim, '+')

legend('Exact Solutions', '', 'Simulation results');
title(desc)
xlabel('time [sec]');
ylabel('Pressure');
grid on;

print -dpng pistonValidation.png
  
pause
unix('/bin/rm press pistonVal.dat');
