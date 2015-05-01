%_________________________________
% 11/03/06   --Todd
% This matLab script generates 2D plots
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
uda = 'impHotBlob.uda'

desc = 'Semi-Implicit hot blob';
desc2 = 'Example: of 2D lineextract';

legendText = {'L-0','L-1', 'L-2', 'L-3', 'L-4'};                   
symbol     = {'+','*r','xg','squarem','diamondb'};
mat        = 0;                 % material index

numPlotCols = 1;               % for multiple plots on 1 page      
numPlotRows = 1;
plotVar1    = true;

% lineExtract Options
startEnd  ={'-istart 1 1 10 -iend 100 100 10'};
varLabel1 = 'delP_Dilatate';

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
%  extract the physical time for each data
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = load('-ascii','tmp');
nDumps = length(physicalTime)

%set(0,'DefaultFigurePosition',[0,0,900,600]);
%_________________________________
% Loop over all the timesteps
for(n = 1:nDumps )
  %n = input('input timestep')         % comment this out for a movie
  ts = n-1;
  
  time = sprintf('%e sec',physicalTime(n));
  
  %find max number of levels at this timestep
  c0 = sprintf('puda -gridstats %s -timesteplow %i -timestephigh %i |grep "Number of levels" | grep -o \\[0-9\\] >& tmp', uda, ts,ts);
  [s, maxLevel]=unix(c0);
  maxLevel  = load('-ascii','tmp');
  levelExists{1} = 0;
   
  % find what levels exists at this timestep
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
      %   varLabel 1
      if plotVar1
        % extract the data to a file
        c6 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o V1.dat -m %i  -uda %s',varLabel1, level,ts,S_E,mat,uda);
        [s6, r6]=unix(c6);
        
        % load teh data into the arrays
        v1{1,L} = load('-ascii','V1.dat');
        x     = v1{1,L}(:,1);
        y     = v1{1,L}(:,2);
        value = v1{1,L}(:,4);
        n_x_cells = length(unique(x));
        n_y_cells = length(unique(y));
        
        z = reshape(value,[n_x_cells n_y_cells]);
       
        subplot(numPlotRows,numPlotCols,plotNum),contour(x(1:n_x_cells),y(1:n_y_cells:end),z,100);
        t = sprintf('%s Time: %s\n%s',desc, time,desc2);
        title(t);
        %axis([0.175 0.225]);
        %axis([2.75 5.75 0.5 1])
        
        ylabel(varLabel1);
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
%  M(n) = getframe(gcf);
end  % timestep loop


%__________________________________
% show the move and make an avi file
%hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',5,'quality',100);
%unix('ffmpeg -qscale 1 -i test.avi -r 30 test.mpg');
%clear all
