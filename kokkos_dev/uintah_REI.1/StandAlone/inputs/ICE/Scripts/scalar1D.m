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

%________________________________
% USER INPUTS
%uda = 'advectScalarAMR-2L_r2-I2.new.uda'
uda = 'impAdvectAMR.uda'

desc = '1D uniform advection of a passive scalar, refinement ratio 2';
desc2 = 'Linear CFI Interpolation, SecondOrder advection, Refluxing off';

legendText = {'L-0','L-1', 'L-2'}
pDir = 1;                    
symbol = {'+-','*-r','xg'}
mat      = 0;

numPlotCols = 1;
numPlotRows = 1;

plotScalarF = true;
plotScalarFaceflux =false;


% lineExtract start and stop
%startEnd ={'-istart 0 -1 0 -iend 0 100 0';'-istart 0 -1 0 -iend 0 100 0' ;'-istart 0 0 0 -iend 200 0 0'};
startEnd ={'-istart -1 0 0 -iend 100 0 0';'-istart -1 0 0 -iend 100 0 0' ;'-istart 0 0 0 -iend 200 0 0'};
%________________________________
%  extract the physical time for each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda)
[status0, result0]=unix(c0);
physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;

set(0,'DefaultFigurePosition',[0,0,900,600]);
%_________________________________
% Loop over all the timesteps
for(ts = 1:nDumps )
  ts = input('input timestep') 
  time = sprintf('%d sec',physicalTime(ts));
  
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
      
      %____________________________
      %   scalar-F
      if plotScalarF
        c6 = sprintf('lineextract -v scalar-f -l %i -cellCoords -timestep %i %s -o scalar-f.dat -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        scalar{1,L} = importdata('scalar-f.dat');
        x = scalar{1,L}(:,pDir);
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,scalar{1,L}(:,4),symbol{L})
        t = sprintf('%s Time: %s\n%s',desc, time,desc2)
        title(t);
        axis([2.75 5.75 0.5 1])
        
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
      %   Reflux correcton
      if plotScalarFaceflux
        c6 = sprintf('lineextract -v scalar-f_X_FC_flux -l %i -cellCoords -timestep %i %s -o scalar-f_X_FC_flux.dat  -m %i  -uda %s',level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        scalar{1,L} = importdata('scalar-f_X_FC_flux.dat');
        x = scalar{1,L}(:,pDir);
        
        if( L == maxLevel)
            tmp = scalar{1,L}(:,4) .* 2;
        else
            tmp = scalar{1,L}(:,4);
        end
        
        subplot(numPlotRows,numPlotCols,plotNum), plot(x,tmp,symbol{L})
        t = sprintf('%s Time: %s \n',desc, time,desc2)
       
        axis([0 10 0. 10])
        
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

    end  % if level exists
  end  % level loop
  M(ts) = getframe(gcf);
end  % timestep loop


%____________________________
%   sum_scalar_f
figure(2)
c = sprintf('%s/sum_scalar_f.dat',uda);
sumscalar = importdata(c)
t = sumscalar(:,1);
plot(t,sumscalar(:,2))
ylabel('sum_scalar_f');
xlabel('time');
t = sprintf('%s\n%s',desc,desc2)
title(t);

%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',5,'quality',100);
%unix('ffmpeg -qscale 1 -i test.avi -r 30 test.mpg');
%clear all
