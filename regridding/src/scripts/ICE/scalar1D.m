%_________________________________
% 10/01/06   --Todd
% This matLab script generates 1D plots
% of a passive scalar
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
uda = 'impAdvectScalarAMR.uda'

desc = '1D uniform advection of a passive scalar, 0:L:1, 1600 cells';
desc2 = 'SecondOrder Advection';

legendText = {'L-0','L-1', 'L-2', 'L-3', 'L-4'};
pDir   = 1;                    
symbol = {'+','*r','xg','squarem','diamondb'};
mat    = 0;

exactSolution = 'exponential' %'linear' %'sinusoidal'; %linear; %cubic; %quad
velocity    = 1.23456;
exactSolMin = 0.0;
exactSolMax = 1.0;
freq  = 1;
slope = 1;
coeff = 100;

numPlotCols = 1;
numPlotRows = 2;

plotScalarF        = true;
plotError          = true;
plotRHS            = false;
plotScalarFaceflux = false;
plotScalarFaceCorr = false;
plotSumScalarF     = false;
plotScalarLimiter  = false


% lineExtract start and stop
startEnd ={'-istart 0 0 0 -iend 10000 0 0';
           '-istart 0 0 0 -iend 400 0 0';
           '-istart 0 0 0 -iend 400 0 0';
           '-istart 0 0 0 -iend 400 0 0';
           '-istart 0 0 0 -iend 400 0 0'};
%________________________________
%  extract the physical time for each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = load('-ascii','tmp');
nDumps = length(physicalTime)

set(0,'DefaultFigurePosition',[0,0,900,600]);
%_________________________________
% Loop over all the timesteps
for(n = 1:nDumps )
  n = input('input timestep') 
  ts = n-1;
  
  time = sprintf('%e sec',physicalTime(n));
  
  %find max number of levels
  c0 = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i |grep "Number of levels" | grep -o \\[0-9\\] >& tmp',uda, ts,ts);
  [s, maxLevel]=unix(c0);
  maxLevel  = load('-ascii','tmp');
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
      
      %____________________________
      %   scalar-F
      if plotScalarF
        c6 = sprintf('lineextract -v scalar-f -l %i -cellCoords -timestep %i %s -o scalar-f.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        scalarArray{1,L} = load('-ascii','scalar-f.dat');
        xx     = scalarArray{1,L}(:,pDir);
        xindx  =-1:length(xx)-2;
        scalar = scalarArray{1,L}(:,4);


        %_________________________________
        % Exact Solution on each level
        dist   = exactSolMax - exactSolMin;
        offset = physicalTime(n) * velocity;
        xmin   = exactSolMin + offset;
        xmax   = exactSolMax + offset;
        uda_dx = xx(2) - xx(1);
        x = xmin:uda_dx:xmax;
        exactSol=xx * 0;
        
        for( i = 1:length(xx))
          if(xx(i) >= xmin && xx(i) <= xmax)
            d = (xx(i) - xmin )/dist;
            
            if( strcmp(exactSolution,'linear'))
              exactSol(i) = exactSol(i) + slope .* d;
            end

            if(strcmp(exactSolution,'sinusoidal'))
              exactSol(i) = exactSol(i) + sin( 2.0 * freq * pi .* d);
            end
            
            if(strcmp(exactSolution,'cubic'))
              if(d <= 0.5)
                exactSol(i) = -1.3333333 *d^3        + d^2;
              else
                exactSol(i) = -1.3333333*(1.0 - d)^3 + (1.0 - d)^2;
              end
            end
     	      if(strcmp(exactSolution,'quad'))
              if(d <= 0.5)
                exactSol(i) = d^2 -d;
              else
                exactSol(i) = (1.0 - d)^2 - (1.0 - d);
              end
            end
            if(strcmp(exactSolution,'exponential'))
                exactSol(i) = coeff * exp(-1.0/( d * ( 1.0 - d ) + 1e-100) );
            end
          end
        end

        difference = scalar - exactSol;
        L2norm     = sqrt( sum(difference.^2)/length(difference) )
        LInfinity  = max(difference)
         
        % plot the results
        subplot(numPlotRows,numPlotCols,plotNum), plot(xx,scalar,symbol{L})
        hold on;
        plot(xx,exactSol,'-r');
        t = sprintf('%s Time: %s\n%s',desc, time,desc2);
        title(t);
        %xlim([200 600]);
        
        ylabel('scalar');
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   error
      if plotError
        subplot(numPlotRows,numPlotCols,plotNum), plot(xx,difference,symbol{L})
        ylabel('error');
        %xlim([200 600]);
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
        rhs{1,L} = load('-ascii','rhs.dat');
        xx = rhs{1,L}(:,pDir);
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(xx,rhs{1,L}(:,4),symbol{L})
        
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      %____________________________
      %   Reflux flux
      if plotScalarFaceflux
        c6 = sprintf('lineextract -v scalar-f_X_FC_flux -l %i -cellCoords -timestep %i %s -o scalar-f_X_FC_flux.dat  -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        scalarFlux{1,L} = load('-ascii','scalar-f_X_FC_flux.dat');
        xx = scalarFlux{1,L}(:,pDir);
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(xx,scalarFlux{1,L}(:,4),symbol{L})
        t = sprintf('%s Time: %s \n',desc, time,desc2);
        
        ylabel('Flux_{scalar}');
        legend(legendText);
        grid on;
        if (L == maxLevel)
         hold off;
        else
         hold on;
        end
        plotNum = plotNum+ 1;
      end
      
      %____________________________
      %   Reflux Correction
      if plotScalarFaceCorr
        if(L~=maxLevel)
          c6 = sprintf('lineextract -v scalar-f_X_FC_corr -l %i -cellCoords -timestep %i %s -o scalar-f_X_FC_corr.dat  -m %i  -uda %s',level,ts,S_E,mat,uda);
          [s6, r6]=unix(c6);
          scalarCorr{1,L} = load('-ascii','scalar-f_X_FC_corr.dat');
          xx = scalarCorr{1,L}(:,pDir);
          normalizedScalarCorr = scalarCorr{1,L}(:,4)./(scalarFlux{1,L}(:,4) + 1e-100);

          subplot(numPlotRows,numPlotCols,plotNum), plot(xx,normalizedScalarCorr,symbol{L})
          t = sprintf('%s Time: %s \n',desc, time,desc2);
          ylim([-1 1]);
          
          ylabel('Normalized reflux correction Flux_{scalar}');
          legend(legendText);
          grid on;
          plotNum = plotNum+ 1;
        end
      end

      %____________________________
      %   Scalar limiter
      if plotScalarLimiter
        f_name = sprintf('limiter/%i.dat',n);
        limiter{1,L} = load('-ascii',f_name);
        xx = limiter{1,L}(:,pDir);
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(xx,limiter{1,L}(:,2),symbol{L})
        %xlim([200 600]);
        ylabel('gradientLimiter_{scalar}');
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
  'sum scalar on coarse level ',sum(scalarArray{1,1}(:,4))
%  M(n) = getframe(gcf);
end  % timestep loop

%____________________________
%   sum_scalar_f
if(plotSumScalarF)
  figure(2)
  c = sprintf('%s/sum_scalar_f.dat',uda);
  sumscalar = load('-ascii',c);
  t = sumscalar(:,1);
  plot(t,sumscalar(:,2))
  ylabel('sum_scalar_f');
  xlabel('time');
  t = sprintf('%s\n%s',desc,desc2)
  title(t);
end

%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',5,'quality',100);
%unix('ffmpeg -qscale 1 -i test.avi -r 30 test.mpg');
%clear all
