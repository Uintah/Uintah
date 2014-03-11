%____________________________________________
% pistonVal_Solution.m
% This script generates the exact solution to the
% piston validation problem.  It then dumps it to
% a file and also compare the data to an uda file
%____________________________________________

clear all;
close all;
setenv('LD_LIBRARY_PATH', ['/usr/lib']);

%_____________________________________________________________________
% Problem specific variables
uda           = sprintf('imp_pistonVal.uda');    % uda to compare against
piston_vel    = 0.001;
piston_height = 0.001;
delta_y        = 0.0025./100.0;           % domain length/# cells
h_initial1     = piston_height-delta_y;   % initial height of the piston.
h_initial2     = piston_height;
t_final        = 0.8;                     % when the piston stops
gamma          = 1.4;   
p_initial      = 1.01325;                 % initial pressure in chamber
delT_dump      = 1e-2;                    % how often data is dumped
desc = 'Piston Validation problem, Pressure probe at the bottom of the cylinder';
startEnd = '-istart 0 0 0 -iend 0 100 0'  % lineExtract start and stop

%_____________________________________________________________________
% exact solution
% gamma law for the gas.
% P_chamber = p_initial(Vol_initial/Vol_t) ^ gamma
% Since we don't know where the exact surface is we compute
% an upper and lower bound.

time_sec=[0:delT_dump:t_final];

tmp = (h_initial1./(h_initial1-(piston_vel*time_sec)));
p_chamber_1 = p_initial*tmp.^gamma;

tmp = (h_initial2./(h_initial1-(piston_vel*time_sec)));
p_chamber_2 = p_initial*tmp.^gamma;
  
% plot the chamber pressure
% figure
% plot(time_sec, p_chamber_1, time_sec, p_chamber_2)
% xlabel('time')
% ylabel('Pressure')
% grid on;

%_____________________
%Write out the exact solution
file_id=fopen('pistonVal.dat','w');
fprintf(file_id,'%6.4f %15.3f  %15.3f\n',[time_sec;p_chamber_1;p_chamber_2]);
fclose(file_id);

%________________________________
% do the Uintah utilities exist?
[s0, r0]=unix('puda');
[s1, r1]=unix('lineextract');
if( s0 ~=0 || s1 ~= 0)
  disp('Cannot execute uintah utilites puda or lineextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
  return;
end
%________________________________
%  extract the physical time of each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;

%_________________________________
% Loop over all the timesteps
fprintf ('Extracting data from timestep');
for( ts = 1:nDumps )  
  fprintf('%i ',ts);
  c1 = sprintf('lineextract -v press_CC -timestep %i %s -o press -m 0 -uda %s',ts,startEnd,uda);
  [status1, result1]=unix(c1);
  pressArray  = importdata('press');
  
  p_probe(ts)     = pressArray(1:1,4);     %extract pressure at probe index 1
  time_probe(ts)  = physicalTime(ts);      %time when measurement was made
end
  
%________________________________
% plot up the the exact solution and 
% simulation results
figure
plot(time_sec, p_chamber_1, time_sec, p_chamber_2, time_probe, p_probe, '.')
legend('Exact Solutions', '', 'Simulation results');
title(desc)
xlabel('time [sec]');
ylabel('Pressure');
grid on;
  
unix('/bin/rm press pistonVal.dat');
