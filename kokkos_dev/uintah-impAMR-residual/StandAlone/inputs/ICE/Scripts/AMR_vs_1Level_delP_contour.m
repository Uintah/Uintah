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
startEnd ='-istart -1 -1 8 -iend 100 100 8';

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
    X = reshape(x1L, [18 18]);
    Y = reshape(y1L, [18 18]);
    Z = reshape((z2L - z1L), [18 18]);
  
    [C,h] = contourf(X, Y ,Z);
    clabel(C,h);
    colormap jet
    
    xlabel('x')
    ylabel('y')

    grid on;
    if (L == maxLevel)
        hold off;
    else
        hold on;
    end
    filename = sprintf('%i.jpg',ts);
    saveas(gcf,filename,'jpg');
    %M(ts) = getframe(gcf);
end  % timestep loop

%montage -geometry "640x480" -tile 3x3 1.jpg 2.jpg 3.jpg 4.jpg 5.jpg 6.jpg 7.jpg 8.jpg 9.jpg montage.jpg

%clear all
