%_________________________________
% 09/01/04   --Todd
% This matLab script generates 1D plots
% of piston validation data and makes a movie
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
udas = {'imp_pistonVal2.uda.001';'imp_pistonVal.uda.001'};
%udas = cellstr(udas);
desc = 'hack vs stock pistonVal,mpmice';
legendText = {'hack','stock'}
pDir = 2;                    
symbol = {'+','*r'}

% lineExtract start and stop
startEnd = '-istart 0 0 0 -iend 0 100 0 -m 1'

%________________________________
% do the Uintah utilities exist
[s0, r0]=unix('puda');
[s1, r1]=unix('lineextract');
if( s0 ~=0 || s1 ~= 0)
  disp('Cannot execute uintah utilites puda or lineextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
  return;
end
%________________________________
%  extract the physical time for each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',udas{1:1})
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;

  
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%_________________________________
% Loop over all the timesteps
for(ts = 1:nDumps )
  ts = input('input timestep') 
  time = sprintf('%d sec',physicalTime(ts+1));
  
  for(i = 1:2)      %  loop over both udas
    i
    uda = udas{i}
    %use line extract to pull out the data
    c1 = sprintf('lineextract -v rho_CC   -timestep %i %s -o rho.dat      -uda %s',ts,startEnd,uda);
    c2 = sprintf('lineextract -v vel_CC   -timestep %i %s -o vel_tmp.dat  -uda %s',ts,startEnd,uda);
    c3 = sprintf('lineextract -v temp_CC  -timestep %i %s -o temp.dat     -uda %s',ts,startEnd,uda);
    c4 = sprintf('lineextract -v press_CC -timestep %i %s -o press.dat    -m 0   -uda %s',ts,startEnd,uda);
    c5 = sprintf('lineextract -v sp_vol_CC     -timestep %i %s -o sp_vol.dat     -uda %s',ts,startEnd,uda);
    c6 = sprintf('lineextract -v vol_frac_CC   -timestep %i %s -o vol_frac.dat   -uda %s',ts,startEnd,uda);
    c7 = sprintf('lineextract -v speedSound_CC -timestep %i %s -o speedSound.dat -uda %s',ts,startEnd,uda);
    
    [status1, result1]=unix(c1);
    [status2, result2]=unix(c2);
    [status3, result3]=unix(c3);
    [status4, result4]=unix(c4);
    [status5, result5]=unix(c5);
    [status6, result6]=unix(c6);
    [status7, result7]=unix(c7);

    % rip out [ ] from velocity data
    c7 = sprintf('sed ''s/\\[//g'' vel_tmp.dat | sed ''s/\\]//g'' >vel.dat');
    [status7, result7]=unix(c7);

    % import the data into arrays
    press1{1,i}  = importdata('press.dat');
    temp1{1,i}   = importdata('temp.dat');
    rho1{1,i}    = importdata('rho.dat');
    vel1{1,i}    = importdata('vel.dat');
    f1{1,i}      = importdata('sp_vol.dat');
    vf{1,i}      = importdata('vol_frac.dat');
    c{1,i}       = importdata('speedSound.dat');

    unix('/bin/rm *.dat');

    %__________________________________________________
    %  Now plot it up
    %__________________________________________________
    %  temperature
    subplot(2,4,1), plot(temp1{1,i}(:,pDir),temp1{1,i}(:,4),symbol{i})
    xlim ([0 50])
    %axis([0 5 295 320])
    xlabel('x')
    ylabel('temp')
    title(time);
    legend(legendText{1}, legendText{2})
    grid on;
    hold;

    %______________________________
    % pressure
    subplot(2,4,2),plot(press1{1,i}(:,pDir),press1{1,i}(:,4),symbol{i})
    xlim ([0 50])
    %axis([0 5 101000 109000])
    xlabel('x')
    ylabel('pressure')
    title(desc);
    grid on;
    hold;

    %_____________________________
    %  Density
    subplot(2,4,3), plot(rho1{1,i}(:,pDir),rho1{1,i}(:,4),symbol{i})
    xlim ([0 50])
    %axis([0 5 1.75 1.9])
    xlabel('x')
    ylabel('rho')
    grid on;
    hold;

    %____________________________
    %   sp_vol
    subplot(2,4,4), plot(f1{1,i}(:,pDir),f1{1,i}(:,4),symbol{i})
    xlim ([0 50])
    %axis([0 50 0 0.01])
    ylabel('sp vol');
    grid on;
    hold;

    %____________________________
    %  velocity
    subplot(2,4,5), plot(vel1{1,i}(:,pDir), vel1{1,i}(:,5),symbol{i})
    legend('y');
    xlim ([0 50])
    %axis([0 5 -10 10])
    ylabel('vel CC');
    grid on;
    hold;
  
    %____________________________
    %   vol_frac
    subplot(2,4,6), plot(vf{1,i}(:,pDir),vf{1,i}(:,4),symbol{i})
    xlim ([0 50])
    %axis([0 5 -10 10])
    ylabel('vol frac');
    grid on;
    hold;
  
    %____________________________
    %   speedSound
    subplot(2,4,7), plot(c{1,i}(:,pDir),c{1,i}(:,4),symbol{i})
    xlim ([0 50])
    %axis([0 5 -10 10])
    ylabel('speedSound');
    grid on;
    hold;
 end
end
%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'delT.avi','fps',5,'quality',100);
%clear all
