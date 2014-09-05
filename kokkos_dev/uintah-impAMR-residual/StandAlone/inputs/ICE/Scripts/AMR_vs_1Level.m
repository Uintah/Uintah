%_________________________________
% 08/01/05   --Todd
% This script compares AMR shock tube results with
% single level results
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
uda = 'impAdvect_2Level.uda.000'
singleLevelUdas = {'impAdvect_1Level.uda.000'}


desc = 'Single level vs multi-level coarse grid';
legendText = {'Single level solution','Multi-level Solution','Single level solution, fine','L-1', 'Single level solution, finest','L-2'}
pDir = 1;                    
symbol = {'+r','*r','xg'}
symbol2 = {'--b','r','-.g'}
mat    = 0;

numPlotCols = 1;
numPlotRows = 1;

plotRho   =false;
plotTemp  = false;           
plotPress      = true;
plotVel        = false; 
plotRefineFlag = false;
% lineExtract start and stop
startEnd ={'-istart 0 0 0 -iend 100 0 0';'-istart 0 0 0 -iend 200 0 0' ;'-istart 0 0 0 -iend 1000 0 0'};

%________________________________
%  extract the physical time for each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda)
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;
  
%set(0,'DefaultFigurePosition',[0,0,640,480]);
%_________________________________
% Loop over all the timesteps
for(ts = 1:nDumps )
  ts = input('input timestep') 
  time = sprintf('%d sec',physicalTime(ts+1));
  
  %______________________________
  L = 1;
  level = L-1;
  maxLevel = L;

  S_E = startEnd{L};
  unix('/bin/rm -f press temp temp_a rho vel sp_vol');
  plotNum = 1;


  %______________________________THIS IS GROSS AND NEEDS TO BE
  %ENCAPSULATED
  %  temperature   
  if plotTemp
    clear temp1;
    c1  = sprintf('lineextract -v temp_CC -l %i -cellCoords -timestep %i %s -o temp   -m %i  -uda %s',level,ts,S_E,mat,uda);
    c1a = sprintf('lineextract -v temp_CC -l 0 -cellCoords -timestep %i %s -o temp_a -m %i  -uda %s',ts,S_E,mat,singleLevelUdas{L});
    [s1, r1]   = unix(c1);
    [s1a, r1a] = unix(c1a);
    temp1{1,L} = importdata('temp');
    temp_a     = importdata('temp_a');

    x = temp1{1,L}(:,pDir)
    subplot(numPlotRows,numPlotCols,plotNum), plot(temp_a(:,1),temp_a(:,4),symbol2{L},x,temp1{1,L}(:,4),symbol{L})
%     xlim([0.3 0.9])
%     ylim([200 450])
    legend(legendText);
    xlabel('x')
    ylabel('Temperature')
    title(desc);
    grid on;
    if (L == maxLevel)
     hold off;
    else
     hold on;
    end
    plotNum = plotNum+ 1;
  end
  %______________________________
  % pressure
  if plotPress
    c2  = sprintf('lineextract -v delP_Dilatate -l %i -cellCoords -timestep %i %s -o press   -m 0 -uda %s',level,ts,S_E,uda);
    c2a = sprintf('lineextract -v delP_Dilatate -l 0  -cellCoords -timestep %i %s -o press_a -m 0 -uda %s',ts,S_E,singleLevelUdas{L});
    [s2, r2]    = unix(c2);
    [s2a,r2a]   = unix(c2a);
    press1{1,L} = importdata('press');
    press_a     = importdata('press_a');
    x1 = press1{1,L}(:,pDir)
    x_a = press_a(:,pDir)

    subplot(numPlotRows,numPlotCols,plotNum),plot(x_a,press_a(:,4),symbol2{L},x1,press1{1,L}(:,4),symbol{L})
%     xlim([0.3 0.9])
%     ylim([10132.5 101325])
    xlabel('x')
    ylabel('delP_Dilatate')

    grid on;
    if (L == maxLevel)
     hold off;
    else
     hold on;
    end
    plotNum = plotNum+ 1;
  end
  %_____________________________
  %  Density
  if plotRho
    c3  = sprintf('lineextract -v rho_CC -l %i -cellCoords -timestep %i %s -o rho   -m %i  -uda %s',level,ts,S_E,mat,uda);
    c3a = sprintf('lineextract -v rho_CC -l 0  -cellCoords -timestep %i %s -o rho_a -m %i  -uda %s',ts,S_E,mat,singleLevelUdas{L});
    [s3, r3]    = unix(c3);
    [s3a, r3a]  = unix(c3a);
    rho1{1,L} = importdata('rho');
    rho_a     = importdata('rho_a');


    subplot(numPlotRows,numPlotCols,plotNum), plot(rho_a(:,1),rho_a(:,4),symbol2{L},x,rho1{1,L}(:,4),symbol{L})
%     xlim([0.3 0.9])
%     ylim([0 1.3])
    xlabel('x')
    ylabel('Density')
    grid on;
    if (L == maxLevel)
     hold off;
    else
     hold on;
    end
    plotNum = plotNum+ 1;
  end
  %____________________________
  %  velocity
  if plotVel
    c5  = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o vel_tmp    -m %i  -uda %s',level,ts,S_E,mat,uda);
    c5a = sprintf('lineextract -v vel_CC -l 0  -cellCoords -timestep %i %s -o vel_tmp_a -m %i  -uda %s',ts,S_E,mat,singleLevelUdas{L});
    [s5, r5]  =unix(c5);
    [s5a, r5a]=unix(c5a);
    % rip out [ ] from velocity data
    c6  = sprintf('sed ''s/\\[//g'' vel_tmp   | sed ''s/\\]//g'' >vel');
    c6a = sprintf('sed ''s/\\[//g'' vel_tmp_a | sed ''s/\\]//g'' >vela');
    [s6, r6]    = unix(c6);
    [s6a, r6a]  = unix(c6a);
    vel1{1,L} = importdata('vel');
    vel_a     = importdata('vela');

    subplot(numPlotRows,numPlotCols,plotNum), plot(vel_a(:,1),vel_a(:,4),symbol2{L},x, vel1{1,L}(:,4),symbol{L})
    %legend('y');
%     xlim([0.3 0.9])
%     ylim([0.0 350])
    ylabel('Velocity');
    grid on;
    if (L == maxLevel)
     hold off;
    else
     hold on;
    end
    plotNum = plotNum+ 1;
  end
  %____________________________
  %   refineFlag
  if plotRefineFlag
    c6 = sprintf('lineextract -v refineFlag -l %i -cellCoords -timestep %i %s -o refineFlag -m %i  -uda %s',level,ts,S_E,mat,uda);
    [s6, r6]=unix(c6);
    rF{1,L} = importdata('refineFlag');


    subplot(numPlotRows,numPlotCols,plotNum), plot(x,rF{1,L}(:,4),symbol{L})
    %xlim([0.3 0.9])
    %axis([0 5 -10 10])
    ylabel('refineFlag');
    grid on;
    if (L == maxLevel)
     hold off;
    else
     hold on;
    end
    plotNum = plotNum+ 1;
  end

  %M(ts) = getframe(gcf);
end  % timestep loop
%__________________________________
% show the move and make an avi file
hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',30,'quality',100);
%clear all
