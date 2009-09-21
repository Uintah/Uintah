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
setenv('LD_LIBRARY_PATH', ['/usr/lib']);

%________________________________
% USER INPUTS
uda = 'shockTube_AMR.uda.001'
singleLevelUdas = {'shockTube_50.uda.000','shockTube_200.uda.000','shockTube_200.uda.000'}


desc = 'ShockTube: 2 Levels, piecewise constant interpolation, Second Order Advection, Refluxing, Refinement ratio 4:1';
legendText = {'Single level solution, coarse','L-0','Single level solution, fine','L-1', 'Single level solution, finest','L-2'}
pDir = 1;                    
symbol = {'+','*r','xg'}
symbol2 = {'--b','r','-.g'}
mat    = 0;

numPlotCols = 1;
numPlotRows = 4;

plotRho   = true;
plotTemp  = true;           
plotPress      = true;
plotVel        = true; 
plotRefineFlag = false;
% lineExtract start and stop
startEnd ={'-istart 0 0 0 -iend 100 0 0';'-istart 0 0 0 -iend 200 0 0' ;'-istart 0 0 0 -iend 1000 0 0'};

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
  
  %find max number of levels
  c0 = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i |grep "Number of levels" | grep -o \\[0-9\\] >& tmp',uda, ts,ts);
  [s, maxLevel]=unix(c0);
  maxLevel  = importdata('tmp');
  levelExists{1} = 0;
   
  % find what levels exists
  for(L = 2:maxLevel)
    c0 = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i >& tmp ; grep "Level: index %i" tmp',uda, ts,ts, L-1);
    [levelExists{L}, result0]=unix(c0);
  end 
  
  %______________________________
  for(L = 1:maxLevel) 
    level = L-1;
    if (levelExists{L} == 0) 
     
      S_E = startEnd{L};
      unix('/bin/rm -f press temp temp_a rho vel sp_vol');
      plotNum = 1;
      level

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
        xlim([0.3 0.9])
        ylim([200 450])
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
        c2  = sprintf('lineextract -v press_CC -l %i -cellCoords -timestep %i %s -o press   -m 0 -uda %s',level,ts,S_E,uda);
        c2a = sprintf('lineextract -v press_CC -l 0  -cellCoords -timestep %i %s -o press_a -m 0 -uda %s',ts,S_E,singleLevelUdas{L});
        [s2, r2]    = unix(c2);
        [s2a,r2a]   = unix(c2a);
        press1{1,L} = importdata('press');
        press_a     = importdata('press_a');
       
        subplot(numPlotRows,numPlotCols,plotNum),plot(press_a(:,1),press_a(:,4),symbol2{L},x,press1{1,L}(:,4),symbol{L})
        xlim([0.3 0.9])
        ylim([10132.5 101325])
        xlabel('x')
        ylabel('Pressure')
        
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
        xlim([0.3 0.9])
        ylim([0 1.3])
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
        xlim([0.3 0.9])
        ylim([0.0 350])
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
        xlim([0.3 0.9])
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
    end  % if level exists
  end  % level loop
  %M(ts) = getframe(gcf);
end  % timestep loop
%__________________________________
% show the move and make an avi file
hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',30,'quality',100);
%clear all
