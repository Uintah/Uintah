%_________________________________
% 01/28/05   --Todd
% This matLab script generates 1D plots for a
% reflected shock tube Validation problem                                         
% Reference: " A High-resolution TVD Finite Volume Scheme for the    
%             Euler Equations in Conservation Form" J.C.T. Wand     
%             G.F. Widhopf, AIAA 25th Aerospace Science Meeting,    
%             January 12-15, 1987/Reno Nevada, AIAA-97-0538          
% Examine the results at 0.95, 2.25 and 2.75 sec.  compare them to  
% Fig 6(a,b,c) 
%  NOTE:  You must have setenv MATLAB_SHELL "/bin/tcsh -f"
%  in your .cshrc file for this to work
%________________________________
close all;
clear all;
clear function;
setenv('LD_LIBRARY_PATH', ['/usr/lib']);

%________________________________
% USER INPUTS
udas  = {'inferno/shockTube_reflected.uda','inferno/shockTube_reflected_MPMICE.uda'};
matls = {'0','1'};
delX  = {0.02, 0.02}
desc = '';
legendText = {'Pure ICE problem.','MPMICE problem:  MPM material used to cap the right end'}
pDir = 1;                    
symbol = {'.','x-r'}

% lineExtract start and stop
startEnd = '-istart 0 0 0 -iend 1000 0 0'

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
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',udas{1:1})
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;
  
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%_________________________________
% Loop over all the timesteps
k = 0;
for(ts = 1:nDumps )
  ts = input('input timestep') 
  time = sprintf('%d sec',physicalTime(ts+1));
  
  for(i = 1:2)      %  loop over both udas
    i
    uda = udas{i}
    matl = matls{i}
    %use line extract to pull out the data
    c1 = sprintf('lineextract -v rho_CC   -timestep %i %s -o rho     -m %s -uda %s',ts,startEnd,matl,uda);
    c2 = sprintf('lineextract -v vel_CC   -timestep %i %s -o vel_tmp -m %s -uda %s',ts,startEnd,matl,uda);
    c3 = sprintf('lineextract -v temp_CC  -timestep %i %s -o temp    -m %s -uda %s',ts,startEnd,matl,uda);
    c4 = sprintf('lineextract -v press_CC -timestep %i %s -o press   -m 0 -uda %s',ts,startEnd,uda);
    
    [status1, result1]=unix(c1);
    [status2, result2]=unix(c2);
    [status3, result3]=unix(c3);
    [status4, result4]=unix(c4);

    % rip out [ ] from velocity data
    c7 = sprintf('sed ''s/\\[//g'' vel_tmp | sed ''s/\\]//g'' >vel');
    [status7, result7]=unix(c7);

    % import the data into arrays
    press1{1,i}  = importdata('press');
    temp1{1,i}   = importdata('temp');
    rho1{1,i}    = importdata('rho');
    vel1{1,i}    = importdata('vel');

    unix('/bin/rm press temp rho vel sp_vol vol_frac');
    x{i} = press1{1,i}(:,pDir) .* delX{i} - 4.1;
    %__________________________________________________
    %  Now plot it up
    %______________________________
    % pressure
    subplot(3,1,1),plot(x{i},press1{1,i}(:,4),symbol{i})
    axis([-4.2 4.2 0. 6.0])
    set(gca,'XTick',-4.2:0.56:4.2)
    set(gca,'YTick',0:0.6:6.0)
    xlabel('x')
    ylabel('pressure')
    title(time);
    legend(legendText{1}, legendText{2})
    grid on;
    hold;
    %_____________________________
    %  Density
    subplot(3,1,2), plot(x{i},rho1{1,i}(:,4),symbol{i})
    axis([-4.2 4.2 0.0 3.0])
    set(gca,'XTick',-4.2:0.56:4.2)
    set(gca,'YTick',0:0.3:3.0)
    xlabel('x')
    ylabel('rho')
    grid on;
    hold;
    %____________________________
    %  velocity
    subplot(3,1,3), plot(x{i}, vel1{1,i}(:,4),symbol{i})
    axis([-4.2 4.2 -0.6 1.4])
    set(gca,'XTick',-4.2:0.56:4.2)
    set(gca,'YTick',-0.6:0.2:1.4)
    ylabel('vel CC');
    grid on;
    hold;    
  end
  k = k+1
 M(k) = getframe(gcf);
end
%__________________________________
% show the move and make an avi file
hFig = figure;
movie(hFig,M,1,3)
movie2avi(M,'shockTube_reflected.avi','fps',5,'quality',100);
clear all
