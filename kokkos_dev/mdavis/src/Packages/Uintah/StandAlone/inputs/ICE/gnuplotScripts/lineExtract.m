%_________________________________
% 09/01/04   --Todd
% This matLab script generates 1D plots
% from two different uda directories
% using line extract
%
%  NOTE:  You must have 
%  setenv MATLAB_SHELL "/bin/tcsh -f"
%  in your .cshrc file for this to work
%_________________________________

close all;
clear all;
clear function;

%________________________________
% USER INPUTS
uda1 = sprintf('shockTube.uda.001');      %----change this
uda2 = sprintf('shockTube.uda.001');      %----change this
desc     = sprintf('');
uda1_txt = sprintf('');
uda2_txt = sprintf('');
timestep_begin = 1;
timestep_end   = 20;

% lineExtract start and stop
startEnd = '-istart 0 0 0 -iend 100 0 0'
%________________________________



[ans, test]=unix('printenv MATLAB_SHELL');   
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%_________________________________
% Loop over all the timesteps
for(ts = timestep_begin:timestep_end )
  ts = input('input timestep') 
 
  %use line extract to pull out the data
  c1 = sprintf('lineextract -v rho_CC   -timestep %i %s -o rho      -uda %s',ts,startEnd,uda1)
  c2 = sprintf('lineextract -v vel_CC   -timestep %i %s -o vel_tmp  -uda %s',ts,startEnd,uda1)
  c3 = sprintf('lineextract -v temp_CC  -timestep %i %s -o temp      -uda %s',ts,startEnd,uda1)
  c4 = sprintf('lineextract -v press_CC -timestep %i %s -o press     -uda %s',ts,startEnd,uda1)
  c5 = sprintf('lineextract -v delP_Dilatate -timestep %i %s -o delP -uda %s',ts,startEnd,uda1)
  c6 = sprintf('lineextract -v scalar-f -timestep %i %s -o scalarf   -uda %s',ts,startEnd,uda1)
  
  [status1, result1]=unix(c1)
  [status2, result2]=unix(c2)
  [status3, result3]=unix(c3)
  [status4, result4]=unix(c4)
  [status5, result5]=unix(c5)
  [status6, result6]=unix(c6)
    
  % rip out [ ] from velocity data
  c7 = sprintf('sed ''s/\\[//g'' vel_tmp | sed ''s/\\]//g'' >vel')
  [status7, result7]=unix(c7)
  
  
  % import the data into arrays
  press1  = importdata('press');
  temp1   = importdata('temp');
  rho1    = importdata('rho');
  delp1   = importdata('delP');
  vel1    = importdata('vel');
  f1      = importdata('scalarf');
  press2  = importdata('press');
  temp2   = importdata('temp');
  rho2    = importdata('rho');
  delp2   = importdata('delP');
  vel2    = importdata('vel');
  f2      = importdata('scalarf');
  
  %____________________________
  % find principal direction in the data set
  if max(press1(:,1)) ~= min(press1(:,1))
    pDir = 1;
  end
  if max(press1(:,2)) ~= min(press1(:,2)) 
    pDir = 2;
  end
  if max(press1(:,3)) ~= min(press1(:,3))
    pDir = 3;
  end
  %__________________________________________________
  %  Now plot it up
  %__________________________________________________
  %  temperature
  subplot(2,3,1), plot(temp1(:,pDir),temp1(:,4),'-r',temp2(:,pDir),temp2(:,4),'-g')
  legend(uda1_txt, uda2_txt);
  %xlim ([0 5])
  %axis([0 5 295 320])
  xlabel('x')
  ylabel('temp')
  grid on;
  
  %______________________________
  % pressure
  subplot(2,3,2),plot(press1(:,pDir),press1(:,4),'-r',press2(:,pDir),press2(:,4),'-g')
  %xlim ([0 5])
  %axis([0 5 101000 109000])
  xlabel('x')
  ylabel('pressure')
  title(desc);
  grid on;
  
  %_____________________________
  %  Density
  subplot(2,3,3), plot(rho1(:,pDir),rho1(:,4), '-r', rho2(:,pDir),rho2(:,4), '-g')
  %xlim ([0 5])
  %axis([0 5 1.75 1.9])
  xlabel('x')
  ylabel('rho')
  grid on;
  
  %____________________________
  %   delP
  subplot(2,3,4), plot(delp1(:,pDir),delp1(:,4), '-r', delp2(:,pDir), delp2(:,4), '-g')
  %xlim ([0 5])
  %axis([0 5 -10 10])
  ylabel('DelP');
  grid on;
  
  %____________________________
  %  velocity
  subplot(2,3,5), plot(vel1(:,pDir),vel1(:,4), '-r', vel1(:,pDir), vel1(:,5), '-g',vel1(:,pDir),vel2(:,6))
    legend('x','y','z');
  %xlim ([0 5])
  %axis([0 5 -10 10])
  ylabel('vel_CC');
  grid on;
  
  %____________________________
  %  scalar-f
  subplot(2,3,6), plot(f1(:,pDir),f1(:,4), '-r', f2(:,pDir), f2(:,4), '-g')
  %xlim ([0 5])
  %axis([0 5 -10 10])
  ylabel('scalar-f');
  grid on;
  %M(t) = getframe(gcf);
  
  clear press1,press2, temp1, temp2, rho1, rho2,;
  clear delp1, delp2, vel2, vel2;
 
end
%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'delT.avi','fps',5,'quality',100);
%clear all
