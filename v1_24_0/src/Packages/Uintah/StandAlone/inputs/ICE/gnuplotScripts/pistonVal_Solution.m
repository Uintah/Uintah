%____________________________________________
% pistonVal_Solution.m
% This script generates the exact solution to the
% piston validation problem.  It then dumps it to
% a file and also compare the data to an uda file
%____________________________________________

clear all;
close all;
%_____________________________________________________________________
% Problem specific variables
piston_vel    = 0.001;
piston_height = 0.001;
delta_y        = 0.0025./100.0;           % domain length/# cells
h_initial1     = piston_height-delta_y;   % initial height of the piston.
h_initial2     = piston_height;
t_final        = 0.8;                     % when the piston stops
gamma          = 1.4;   
p_initial      = 1.01325;                 % initial pressure in chamber
delT_dump      = 1e-2;                    % how often printdata is dumped
uda            = sprintf('pistonValidation.uda');    % uda to compare against

%_____________________
% gamma law for the gas.
% P_chamber = p_initial(Vol_initial/Vol_t) ^ gamma
time_sec=[0:delT_dump:t_final];

tmp = (h_initial1./(h_initial1-(piston_vel*time_sec)))
p_chamber_1 = p_initial*tmp.^gamma

tmp = (h_initial2./(h_initial1-(piston_vel*time_sec)))
p_chamber_2 = p_initial*tmp.^gamma
  

% plot the chamber pressure
figure
plot(time_sec, p_chamber_1, time_sec, p_chamber_2)
xlabel('time')
ylabel('Pressure')
grid on;

%_____________________
%Write out data files
file_id=fopen('pistonVal.dat','w');
fprintf(file_id,'%6.4f %15.3f  %15.3f\n',[time_sec;p_chamber_1;p_chamber_2]);
fclose(file_id);


%______________________________________________________________________
% compare against simulation
unix('setenv MATLAB_SHELL "/bin/csh -b"')
command1 = sprintf('cd %s; ../inputs/ICE/gnuplotScripts/findTimesteps>&../ts',uda);
unix(command1);                           % 
timesteps = importdata('ts');   % find the printdata dump indicies

%_________________________________
% Loop over all the timesteps
%  - put printData array into pressArray
for( t = 1:length(timesteps) ) 
  here = timesteps(t);
  p  = sprintf('%s/BOT_equilibration/%d/L-0/patch_0/Press_CC_equil', uda,here); 
  tt = sprintf('%s/BOT_equilibration/%d/simTime', uda,here); 
  pressArray     = sortrows(importdata(p),1);    % put printData Array into pressArray
  p_probe(t)     = pressArray(1:1,2);            %extract pressure at probe index 1
  time_probe(t)  = importdata(tt)               %time when measurement was made
end
  

%________________________________
% plot up the comparison
figure
plot(time_sec, p_chamber_1, time_sec, p_chamber_2, time_probe, p_probe, '.')
legend('Exact Solutions', '', 'Simulation results');
grid on;
  
