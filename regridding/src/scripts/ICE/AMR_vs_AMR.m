% %_________________________________
% 12/19/05   --Todd
% This script compares 2 AMR udas
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
uda1 = 'shockTube_MR_Explicit.uda.000';
uda2 = 'shockTube_MR_Implicit.uda.005';


desc = 'ShockTube: 2 Levels (Implicit timestep)';
legendText = {'Explicit L-0','Implicit L-0','Explicit L-1','Implicit L-1', 'Explicit L-2','Implicit L-2'};
pDir = 1;                    
symbol1 = {'+r','*r','xg'};
symbol2 = {'+b','*b','xg'};
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
unix('/bin/rm -f tmp tmp1 tmp2');

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
c  = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp1',uda1);
ca = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp2',uda2);
[s, r]=unix(c);
[sa, ra]=unix(ca);
physicalTime1  = importdata('tmp1');
physicalTime2  = importdata('tmp2');
nDumps = length(physicalTime1) - 1;
  
%set(0,'DefaultFigurePosition',[0,0,640,480]);
%_________________________________
% Loop over all the timesteps
for(ts = 1:nDumps )
  ts1 = input('input timestep (uda1)') 
  ts2 = input('input timestep (uda2)')
  time = sprintf('%d sec',physicalTime1(ts1+1));
  
  %find max number of levels
  c0  = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i |grep "Number of levels" | grep -o \\[0-9\\] >& tmp1',uda1, ts1,ts1);
  c0a = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i |grep "Number of levels" | grep -o \\[0-9\\] >& tmp2',uda2, ts2,ts2);
  [s,  maxLevel1]=unix(c0);
  [sa, maxLevel2]=unix(c0a);
  maxLevel1  = importdata('tmp1');
  maxLevel2  = importdata('tmp2');
  levelExists{1} = 0;
  
  if( maxLevel1 ~= maxLevel2)
    printf( 'CAN NOT COMPARE UDAS.  MUST HAVE SAME NUMBER OF LEVELS')
    exit
  end
   
  % find what levels exists
  for(L = 2:maxLevel1)
    c0 = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i >& tmp ; grep "Level: index %i" tmp',uda1, ts1,ts1, L-1);
    [levelExists{L}, result0]=unix(c0);
  end 
  
  
  
  %______________________________
  for(L = 1:maxLevel1) 
    level = L-1;
    if (levelExists{L} == 0) 
     
      S_E = startEnd{L};
      unix('/bin/rm -f temp1 temp2 press1 press2 vel1 vel2 vel_tmp1 vel_tmp2 rho1 rho2 refineFlag1 refineFlag2');
      plotNum = 1;
      level

      %______________________________THIS IS GROSS AND NEEDS TO BE
      %ENCAPSULATED
      %  temperature   
      if plotTemp
        clear temp1;
        c1  = sprintf('lineextract -v temp_CC -l %i -cellCoords -timestep %i %s -o temp1 -m %i  -uda %s',level,ts1,S_E,mat,uda1);
        c1a = sprintf('lineextract -v temp_CC -l %i -cellCoords -timestep %i %s -o temp2 -m %i  -uda %s',level,ts2,S_E,mat,uda2);
        [s1, r1]   = unix(c1);
        [s1a, r1a] = unix(c1a);
        temp1{1,L} = importdata('temp1');
        temp2{1,L} = importdata('temp2');
        
        x1 = temp1{1,L}(:,pDir);
        x2 = temp2{1,L}(:,pDir);
        subplot(numPlotRows,numPlotCols,plotNum), plot(x1,temp1{1,L}(:,4),symbol1{L},x2,temp2{1,L}(:,4),symbol2{L})
        xlim([0.3 0.9])
        ylim([200 450])
        legend(legendText);
        xlabel('x')
        ylabel('Temperature')
        this=sprintf('%s \n Physical time: Explicit: %e, Implicit: %e ',desc, physicalTime1(ts1+1), physicalTime2(ts2+1));
        title(this);
        grid on;
        if (L == maxLevel1)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %______________________________
      % pressure
      if plotPress
        c2  = sprintf('lineextract -v press_CC -l %i -cellCoords -timestep %i %s -o press1 -m 0 -uda %s',level,ts1,S_E,uda1);
        c2a = sprintf('lineextract -v press_CC -l %i -cellCoords -timestep %i %s -o press2 -m 0 -uda %s',level,ts2,S_E,uda2);
        [s2, r2]    = unix(c2);
        [s2a,r2a]   = unix(c2a);
        press1{1,L} = importdata('press1');
        press2{1,L} = importdata('press2');
       
        subplot(numPlotRows,numPlotCols,plotNum),plot(x1,press1{1,L}(:,4),symbol1{L},x2,press2{1,L}(:,4),symbol2{L})
        xlim([0.3 0.9])
        ylim([10132.5 101325])
        xlabel('x')
        ylabel('Pressure')
        
        grid on;
        if (L == maxLevel1)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %_____________________________
      %  Density
      if plotRho
        c3  = sprintf('lineextract -v rho_CC -l %i -cellCoords -timestep %i %s -o rho1 -m %i  -uda %s',level,ts1,S_E,mat,uda1);
        c3a = sprintf('lineextract -v rho_CC -l %i -cellCoords -timestep %i %s -o rho2 -m %i  -uda %s',level,ts2,S_E,mat,uda2);
        [s3, r3]    = unix(c3);
        [s3a, r3a]  = unix(c3a);
        rho1{1,L} = importdata('rho1');
        rho2{1,L} = importdata('rho2');
        
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x1,rho1{1,L}(:,4),symbol1{L},x2,rho2{1,L}(:,4),symbol2{L})
        xlim([0.3 0.9])
        ylim([0 1.3])
        xlabel('x')
        ylabel('Density')
        grid on;
        if (L == maxLevel1)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %  velocity
      if plotVel
        c5  = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o vel_tmp1 -m %i  -uda %s',level,ts1,S_E,mat,uda1);
        c5a = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o vel_tmp2 -m %i  -uda %s',level,ts2,S_E,mat,uda2);
        [s5, r5]  =unix(c5);
        [s5a, r5a]=unix(c5a);
        % rip out [ ] from velocity data
        c6  = sprintf('sed ''s/\\[//g'' vel_tmp1 | sed ''s/\\]//g'' >vel1');
        c6a = sprintf('sed ''s/\\[//g'' vel_tmp2 | sed ''s/\\]//g'' >vel2');
        [s6, r6]    = unix(c6);
        [s6a, r6a]  = unix(c6a);
        vel1{1,L} = importdata('vel1');
        vel2{1,L} = importdata('vel2');
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x1,vel1{1,L}(:,4),symbol1{L},x2, vel2{1,L}(:,4),symbol2{L})
        %legend('y');
        xlim([0.3 0.9])
        ylim([0.0 350])
        ylabel('Velocity');
        grid on;
        if (L == maxLevel1)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   refineFlag
      if plotRefineFlag
        c6 = sprintf('lineextract -v refineFlag -l %i -cellCoords -timestep %i %s -o refineFlag1 -m %i  -uda %s',level,ts1,S_E,mat,uda1);
        c6 = sprintf('lineextract -v refineFlag -l %i -cellCoords -timestep %i %s -o refineFlag2 -m %i  -uda %s',level,ts2,S_E,mat,uda2);
        [s7, r7]  =unix(c7);
        [s7a, r7a]=unix(c7a);
        rF1{1,L} = importdata('refineFlag1');
        rF2{1,L} = importdata('refineFlag2');
        
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x1,rF1{1,L}(:,4),symbol1{L},x2,rF2{1,L}(:,4),symbol2{L})
        xlim([0.3 0.9])
        %axis([0 5 -10 10])
        ylabel('refineFlag');
        grid on;
        if (L == maxLevel1)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      
      unix('/bin/rm -f temp1 temp2 press1 press2 vel1 vel2 vel_tmp1 vel_tmp2 rho1 rho2 refineFlag1 refineFlag2 tmp tmp1 tmp2');
      
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
