#! /usr/bin/octave -qf
%______________________________________________________________________
%  This script reads a DataAnalysis:controlVolFluxes file and 
% and plots the net fluxes of Q into/out of the control volume and the total
% change in Q in the control volume.  This is 
%
%  Usage checkControlVol.m <path to cv.dat file>
%______________________________________________________________________

format long

uda = sprintf("%s",argv(){1})


data = load(uda);
t       = data(:,1);
totalCV = data(:,2);
xFluxes = data(:,3);
yFluxes = data(:,4);
zFluxes = data(:,5);


% initialize
dt(:,1) = t(1);
changeTotalCV(:,1) = 0;

%__________________________________
% compute the change in Q in cv
for i=2:length(t)
  dt(i,:) = t(i) - t(i-1);
  changeTotalCV(i,:) = totalCV(i) .- totalCV(i-1);
end

%__________________________________
%  Compute the net flux of Q into/out of cv
for i=1:length(xFluxes)
  allFluxes(i,:) = (xFluxes(i) + yFluxes(i) + zFluxes(i) ) * dt(i,:);
end

%__________________________________
%  
%graphics_toolkit('gnuplot')
h = figure('position', [0, 0, 1024, 768])
plot( t,changeTotalCV,'b:o', "markersize", 3,
      t, allFluxes,   'r:+', "markersize", 3)

title(uda)
xlabel("Time[s]")
ylabel("Change in Q")

l = legend('Change in Q in CV', 'net Q faceFluxes * dt' );
set (l, "fontsize",8);
grid on;

pause
print(h , 'controlVol.png', '-dpng');
