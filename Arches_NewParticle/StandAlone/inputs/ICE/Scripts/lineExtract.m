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
%uda = 'impHotBlobAMR.uda'
uda = 'impAdvectScalarAMR.uda'

desc = '1D impHotBlobAMR, cell R=4,1,1';
legendText = {'L-0','L-1', 'L-2'}
pDir   = 1;                    
symbol = {'+','*r','xg'};
mat    = 0;
xlo    = 0.3;
xhi    = 0.5;

numPlotCols = 1;
numPlotRows = 5;

plotRho         = false;
plotGradRho     = false;
plotTemp        = true;
plotGradTemp    = false;
plotPress       = true;
plotGradPress   = true;
plotDelP        = true;
plotGrad_dp_XFC = false;
plotVol_frac    = false;
plotSp_vol      = false;
plotRHS         = true;
plot_A          = true;
plotVel         = false; 
plotRefineFlag  = false;
% lineExtract start and stop
startEnd ={'-istart -1 0 0 -iend 100 0 0';'-istart 0 0 0 -iend 200 0 0' ;'-istart 0 0 0 -iend 200 0 0'};

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


if(plot_A)
    figure
    figure
end
  
%set(0,'DefaultFigurePosition',[0,0,640,480]);
%_________________________________
% Loop over all the timesteps
for(ts = 0:nDumps )
  ts = input('input timestep ') 
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
      unix('/bin/rm -f *.dat');
      plotNum = 1;
      level
      figure(1);
      
      if (L == maxLevel)
          on_off = 'off';
      else
          on_off =  'on';
      end
      
      %______________________________THIS IS GROSS AND NEEDS TO BE
      %ENCAPSULATED
      %  temperature   
      if plotTemp
        clear temp1;
        c1 = sprintf('lineextract -v temp_CC -l %i -cellCoords -timestep %i %s -o temp.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s1, r1]   = unix(c1);
        temp1{1,L} = importdata('temp.dat');
        
        x = temp1{1,L}(:,pDir);
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,temp1{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        %legend(legendText);
        xlabel('x')
        ylabel('temp')
        title(time);
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %______________________________
      % gradient of temperature
      if plotGradTemp
        dx = abs(x(1) - x(2));
        gradTemp = gradient(temp1{1,L}(:,4), dx);
        
        subplot(numPlotRows,numPlotCols,plotNum),plot(x,gradTemp,symbol{L})
        xlim([xlo xhi]);
        
        xlabel('x')
        ylabel('gradTempCC_x')
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
        c2 = sprintf('lineextract -v press_CC -l %i -cellCoords -timestep %i %s -o press.dat -m 0 -uda %s',level,ts,S_E,uda);
        [s2, r2]    = unix(c2);
        press1{1,L} = importdata('press.dat');
        x = press1{1,L}(:,pDir);
        subplot(numPlotRows,numPlotCols,plotNum),plot(x,press1{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        xlabel('x')
        ylabel('pressure')
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
      % gradient of pressure
      if plotGradPress
        dx = abs(x(1) - x(2));
        gradP = gradient(press1{1,L}(:,4), dx);
        
        subplot(numPlotRows,numPlotCols,plotNum),plot(x,gradP,symbol{L})
        xlim([xlo xhi]);
        
        xlabel('x')
        ylabel('gradPressCC_x')
        title(desc);
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
        c3 = sprintf('lineextract -v rho_CC -l %i -cellCoords -timestep %i %s -o rho.dat  -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s3, r3]  = unix(c3);
        rho1{1,L} = importdata('rho.dat');
        x = rho1{1,L}(:,pDir);
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,rho1{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        xlabel('x')
        ylabel('rho')
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %______________________________
      % gradient of density
      if plotGradRho
        dx = abs(x(1) - x(2));
        gradRho = gradient(rho1{1,L}(:,4), dx);
        
        subplot(numPlotRows,numPlotCols,plotNum),plot(x,gradRho,symbol{L})
        xlim([xlo xhi]);
        
        xlabel('x')
        ylabel('gradRhoCC_x')
        title(desc);
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   sp_vol
      if plotSp_vol
        c4 = sprintf('lineextract -v sp_vol_CC -l %i -cellCoords -timestep %i %s -o sp_vol.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s4, r4] = unix(c4);
        f1{1,L}  = importdata('sp_vol.dat');
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,f1{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        ylabel('sp vol');
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
        c5 = sprintf('lineextract -v vel_CC      -l %i -cellCoords -timestep %i %s -o vel_tmp.dat  -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s5, r5]=unix(c5);
        % rip out [ ] from velocity data
        c6 = sprintf('sed ''s/\\[//g'' vel_tmp.dat | sed ''s/\\]//g'' >vel.dat');
        [s6, r6]  = unix(c6);
        vel1{1,L} = importdata('vel.dat');
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x, vel1{1,L}(:,4),symbol{L})
        %legend('y');
        xlim([xlo xhi]);
        
        ylabel('vel CC');
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   vol_frac
      if plotVol_frac
        c6 = sprintf('lineextract -v vol_frac_CC -l %i -cellCoords -timestep %i %s -o vol_frac.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        vf{1,L} = importdata('vol_frac.dat');
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,vf{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        ylabel('vol frac');
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   delPDilitate
      if plotDelP
        c6 = sprintf('lineextract -v delP_Dilatate -l %i -cellCoords -timestep %i %s -o delP.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        delP{1,L} = importdata('delP.dat');
        x = delP{1,L}(:,pDir);
        
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,delP{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        ylabel('delP');
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   grad_dp_XFC
      if plotGrad_dp_XFC

        c6 = sprintf('lineextract -v grad_dp_XFC -l %i -cellCoords -timestep %i %s -o grad_dp.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        grad_dp{1,L} = importdata('grad_dp.dat');
        x = grad_dp{1,L}(:,pDir);
        
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,grad_dp{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        ylabel('grad_dp');
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   rhs
      if plotRHS
        c6 = sprintf('lineextract -v rhs -l %i -cellCoords -timestep %i %s -o rhs.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        rhs{1,L} = importdata('rhs.dat');
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,rhs{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        ylabel('RHS');
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
        c6 = sprintf('lineextract -v refineFlag -l %i -cellCoords -timestep %i %s -o refineFlag.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        rF{1,L} = importdata('refineFlag.dat');
        
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,rF{1,L}(:,4),symbol{L})
        xlim([xlo xhi]);
        
        ylabel('refineFlag');
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   matrix
      if plot_A
        figure(2);
        c6 = sprintf('lineextract -v matrix -l %i -cellCoords -timestep %i %s -o matrix.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        c6 = sprintf(' cp matrix.dat matrix.tmp.dat; sed s/"A..:"/""/g < matrix.tmp.dat > matrix.dat; rm matrix.tmp.dat');
        [s6, r6]=unix(c6);
        A{1,L} = importdata('matrix.dat');
        
        subplot(7,1,1), plot(x,A{1,L}(:,4),symbol{L}); ylabel('A.p');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end;
        xlim([xlo xhi]); grid on;
        subplot(7,1,2), plot(x,A{1,L}(:,5),symbol{L}); ylabel('A.w');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end
        xlim([xlo xhi]); grid on;
        subplot(7,1,3), plot(x,A{1,L}(:,6),symbol{L}); ylabel('A.e');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end
        xlim([xlo xhi]); grid on;
        subplot(7,1,4), plot(x,A{1,L}(:,7),symbol{L}); ylabel('A.s');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end
        xlim([xlo xhi]); grid on;
        subplot(7,1,5), plot(x,A{1,L}(:,8),symbol{L}); ylabel('A.n');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end
        xlim([xlo xhi]); grid on;
        subplot(7,1,6), plot(x,A{1,L}(:,9),symbol{L}); ylabel('A.b');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end
        xlim([xlo xhi]); grid on;
        subplot(7,1,7), plot(x,A{1,L}(:,10),symbol{L}); ylabel('A.t');
        if (L == maxLevel)
            hold off;
        else
            hold on;
        end
        xlim([xlo xhi]);
        
        grid on;

      end      
      
      
      
    end  % if level exists
  end  % level loop
  %M(ts) = getframe(gcf);
end  % timestep loop
%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',30,'quality',100);
%clear all
