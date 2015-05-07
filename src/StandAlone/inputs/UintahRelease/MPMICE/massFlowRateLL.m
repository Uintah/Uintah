%_________________________________
% 09/01/04   --Todd
% This matLab script generates  a mass flow rate vs time
%  plot.  Make sure you change the user inputs section
%
%  NOTE:  You must have 
%  setenv MATLAB_SHELL "/bin/tcsh -f"
%  in your .cshrc file for this to work
%_________________________________

close all;
clear all;
clear function;
setenv('LD_LIBRARY_PATH', ['/usr/lib']);

%________________________________
% USER INPUTS
uda1   = sprintf('LizardLung4.uda');      %----change this
desc   = sprintf('MassFlow rate through nozzle');
cellArea = 2.0e-6;        

% lineExtract start and stop index
startEnd = '-istart 114 108 0 -iend 114 116 0 -m 2' % top
%startEnd = '-istart 149 33 0 -iend 149 41 0 -m 2'  % bottom

%________________________________
% do the Uintah utilities exist
[s0, r0]=unix('puda');
[s1, r1]=unix('lineextract');
if( s0 ~=0 || s1 ~= 0)
  disp('Cannot execute uintah utilites puda or lineextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
end

%________________________________
%  extract the physical time for each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda1);
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1

%_________________________________
% Loop over all the timesteps
fprintf('working on timestep\n');

for(ts = 1:nDumps )
  fprintf('%i \n',ts);

  %use line extract to pull out the data
  c1 = sprintf('lineextract -v rho_CC   -timestep %i %s -o rho      -uda %s',ts,startEnd,uda1);
  c2 = sprintf('lineextract -v vel_CC   -timestep %i %s -o vel_tmp  -uda %s',ts,startEnd,uda1);
  [status1, result1]=unix(c1);
  [status2, result2]=unix(c2);

  % rip out [] from velocity data
  c3 = sprintf('sed ''s/\\[//g'' vel_tmp | sed ''s/\\]//g'' >vel');
  [status3, result3]=unix(c3);
  
  % import the data into arrays
  rho1    = importdata('rho');
  vel1    = importdata('vel',' ');

  % integrate over the area
  massFlow_tmp = 0.0; 
  u = vel1(:,4);     % form the velocity components
  v = vel1(:,5);
  w = vel1(:,6);

  vel_mag = u;
  massFlow_tmp = (rho1(:,4) .* vel_mag) * cellArea;
  massFlow(ts)= sum(massFlow_tmp);
  
  clear rho1 vel1 massFlow_tmp x y z;
end  

  time_mF=[transpose(physicalTime(1:nDumps));massFlow];
  time_mF=transpose(time_mF);

  save top.txt time_mF -ASCII
 
  %__________________________________________________
  %  Now plot it up
  set(0,'DefaultFigurePosition',[0,0,1024,768]);
  plot(physicalTime(1:nDumps),massFlow,'+r-')
  title(desc);
  xlabel('Time[sec]')
  ylabel('Mass Flow Rate [kg/m^3]')
  grid on;
