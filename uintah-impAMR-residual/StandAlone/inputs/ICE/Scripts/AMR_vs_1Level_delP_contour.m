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
uda_2Level = 'impAdvect_2Level.uda.000'
uda_1Level = 'impAdvect_1Level.uda.000'

desc = 'Single level vs multi-level coarse grid';
legendText = {'Single level solution','Multi-level Solution'}                 
symbol = {'+r','*r','xg'}
symbol2 = {'--b','r','-.g'}

% lineExtract start and stop
startEnd ='-istart 0 0 2 -iend 100 100 2';

%________________________________
%  extract the physical time for each dump
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda_1Level)
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
    L = 1;    % hardwired
    level = L-1;
    maxLevel = L;

    unix('/bin/rm -f press1L press2L');

    c2  = sprintf('lineextract -v delP_Dilatate -l %i  -timestep %i %s -o press2L -m 0 -uda %s',level,ts,startEnd,uda_2Level);
    c2a = sprintf('lineextract -v delP_Dilatate -l 0   -timestep %i %s -o press1L -m 0 -uda %s',      ts,startEnd,uda_1Level);
    [s2, r2]    = unix(c2);
    [s2a,r2a]   = unix(c2a);
    press1L = importdata('press1L');
    press2L = importdata('press2L');
    x1L = press1L(:,1);
    y1L = press1L(:,2);     %level 1 data
    z1L = press1L(:,4);

    x2L = press2L(:,1);
    y2L = press2L(:,2);     % multi level data
    z2L = press2L(:,4);

    
    % Oren --------------------------
    % I have all the data in the vectors. How to I make a contour
    % plot with them?
    %[X, Y] = meshgrid(x2L,y2L);
    %Z = meshgrid(z2L);
    %contour(X,Y,Z)
    stem3(x1L, y1L,z2L)
    xlabel('x')
    ylabel('y')

    grid on;
    if (L == maxLevel)
        hold off;
    else
        hold on;
    end
    %M(ts) = getframe(gcf);
end  % timestep loop
%__________________________________
% show the move and make an avi file
hFig = figure;
%movie(hFig,M,1,3)
%movie2avi(M,'test.avi','fps',30,'quality',100);
%clear all
