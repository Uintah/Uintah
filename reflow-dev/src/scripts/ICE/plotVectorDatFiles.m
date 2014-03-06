#! /usr/bin/octave -qf

%______________________________________________________________________
%  Octave script that plots the magnitude of the vector of multiple dat files
%
%   Usage
%      plotVectorDatFile.m <dat1 dat2 dat3 dat4> <y/n for hardcopy>
%______________________________________________________________________

close all;
clear all;
clear function;

%__________________________________
% generate a hardcopy
function hardcopy(h,filename)

  printf('Generating the hard copy %s', filename);
  FN = findall(h, '-property','FontName');
  FS = findall(h, '-property','FontSize');
  
  set(FN,'FontName','Times');
  set(FS,'FontSize',12);
  saveas( h, filename, "pdf")
  
endfunction

%______________________________________________________________________
 
inputs = argv();
nargs = length(inputs);

datFiles = inputs( 1:nargs );

% if the last input argument == "y" then save the plot
lastEntry = inputs{ nargs };

savePlot = "false";
if ( strcmp( lastEntry, "y" ) )
  savePlot = "true";
  datFiles = inputs( 1:nargs-1 );
endif

datFiles

%define plot symbols
syms = {'r+', 'bo','c*','ys','b.','r^'};


%__________________________________
% loop over all the data files
for j = 1:length(datFiles)

  printf( 'Plotting %s',datFiles{j} )

  % Read in the data file:
  me = sprintf( '%s', char(datFiles{j}) );
  
  % remove brackets [] from the data file 
  c = sprintf( "cat %s | tr -d ''[]'' > .dat", me);
  [s, r] = unix(c);

  % import the data into arrays
  data  = load( '.dat' );
  t = data(:,1);
  x = data(:,2);     % individual components
  y = data(:,3);
  z = data(:,4);

  mag = sqrt(x.^2 + y.^2 + z.^2);
  
  %__________________________________
  %  plot magnitude
  h = figure(1, 'DefaultFigurePosition',[0,0,1024,768]);

  plot( t, mag, syms{j});
  
  legend(datFiles)
  xlabel('Time[sec]')
  ylabel('Magnitude')
  grid on;
  hold on;

  %__________________________________
  % cleanup
  unix( 'rm .dat');
end

if( strcmp(savePlot, "true") )
  unix( 'rm plotVectorData.pdf' );
  hardcopy( h, 'plotVectorData.pdf')
endif 
pause
