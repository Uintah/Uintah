#!/usr/bin/octave -qf
pkg load signal    %  needed for downsample

%______________________________________________________________________
% This postprocessing script reads in data from the DataAnalysis:momentumAnalysis module
% and calculates the forces on the the system.  

% The file is assumed to have the following format:
% # Definitions:
% #    totalCVMomentum:  the total momentum in the control volume at that instant in time
% #    netFlux:          the net flux of momentum through the control surfaces
% #Time                      totalCVMomentum.x()         totalCVMomentum.y()         totalCVMomentum.z()         netFlux.x()                  netFlux.y()                  netFlux.z()
% 0.000000000000000E+00,      1.827906773743646E+01,       1.827906773743646E+01,       1.827906773743646E+01,      3.558930927738402E-12,         3.556266392479301E-12,       3.556266392479301E-12
% 1.000000000000000E-04,      1.827906773742449E+01,       1.827906773742449E+01,       1.827906773742449E+01,      -8.138619200792618E-08,       -8.138619200792618E-08,       -8.138619200792618E-08
% 3.300003189645604E-04,      1.827906773743154E+01,       1.827906773743154E+01,       1.827906773743154E+01,      -3.498177267857727E-07,       -3.498177267857727E-07,       -3.498177258975943E-07
% 5.600006379291208E-04,      1.827906773750070E+01,       1.827906773750069E+01,       1.827906773750069E+01,      -8.050327036102090E-07,       -8.050327053865658E-07,       -8.050327036102090E-07
% 7.900009568936813E-04,      1.827906773767498E+01,       1.827906773767497E+01,       1.827906773767497E+01,      -1.446773100433063E-06,       -1.446773100433063E-06,       -1.446773095103993E-06
% 1.020001275858242E-03,      1.827906773799733E+01,       1.827906773799733E+01,       1.827906773799733E+01,      -2.274781136790693E-06,       -2.274781136790693E-06,       -2.274781135014337E-06

% Below are inputfile specifications:
%     <DataAnalysis>
%       <Module name = "momentumAnalysis">
%         <materialIndex> 1 </materialIndex>          <<< user defined
%         <uvel_FC  label = "uvel_FCME"/>     
%         <vvel_FC  label = "vvel_FCME"/>     
%         <wvel_FC  label = "wvel_FCME"/>     
%         <vel_CC   label = "vel_CC"/>        
%         <rho_CC   label = "rho_CC"/>        
%         
%         <samplingFrequency> 1e4 </samplingFrequency>
%         <timeStart>          0   </timeStart>
%         <timeStop>          100  </timeStop>
%         <controlVolume>
%           <Face side = "x+"  extents = "entireFace"/>
%           <Face side = "x-"  extents = "entireFace"/>
%           <Face side = "y+"  extents = "entireFace"/>
%           <Face side = "y-"  extents = "entireFace"/>
%           <Face side = "z+"  extents = "entireFace"/>
%           <Face side = "z-"  extents = "entireFace"/>
%         </controlVolume>
%         
%       </Module>
%     </DataAnalysis>
%______________________________________________________________________


clear all; clc;
close all;

%______________________________________________________________________
%                HELPER FUNCTIONS
%______________________________________________________________________

%__________________________________
% compute a moving average of the quantity Q
function [q_ave] = moveAve(Q, windowsize)

 q_tmp(:,1) = filter( ones(windowsize,1)/windowsize, 1, Q(:,1) );  % X-dir
 q_tmp(:,2) = filter( ones(windowsize,1)/windowsize, 1, Q(:,2) );  % Y
 q_tmp(:,3) = filter( ones(windowsize,1)/windowsize, 1, Q(:,3) );  % Z

 % trim off the first "windowsize" data points.  They're junk
 lo = windowsize;
 hi = length( q_tmp(:,1) );
 q_ave = q_tmp( lo:hi, 1:3 );

endfunction


%__________________________________
% find lower and upper indicies
function [lo, hi] = trim( t, tLo, tHi)

  lo = min( find( t>=tLo ) );
  hi = max( find( t<=tHi ) );
endfunction

%__________________________________
% generate a hardcopy
function hardcopy(h,filename, pausePlot)

  FN = findall(h,'-property','FontName');
  set(FN,'FontName','Times');
  FS = findall(h, '-property','FontSize');
  set(FS,'FontSize',12);
  saveas( h, filename, "jpg")

  if( strcmp(pausePlot, "true" ) )
    pause
  endif

endfunction

%______________________________________________________________________
%   USER INPUTS
%______________________________________________________________________

%arg_list = argv ()           % to be filled in

me        = "advect_mpmice.uda/momentumAnalysis.dat"
baseTitle = "debugging "

pausePlot = "true";

%______________________________________________________________________
%    DEFAULT VALUES
%______________________________________________________________________
nPoint = 1;                     % Down sampling:  include every n data point
window = 50;                    % moving average window size

tmin =  0.01;                   % time range min
tmax =  0.3;                    % time range max

format long e


%______________________________________________________________________
%     MAIN  
%______________________________________________________________________
%  Load the data into arrays and compute dM/dt where M is the total momentum
%  in the control volume
data = dlmread( me, ",", 4,0 );

%__________________________________
% downsample the data with every nth element
t           = downsample( data(:,1),   nPoint );  % column 1
Mom_cv      = downsample( data(:,2:4), nPoint );  % columns 2 - 4
Mom_netFlux = downsample( data(:,5:7), nPoint );  % columns 5 - 7

%__________________________________
%  Allow user to trim data between tMin and tMax
tmax    = min( max(t), tmax);
[lo,hi] = trim( t, tmin, tmax )

wlo  = lo-window         % subtract off the window size moving average creates needs them for intermediate values
wlo = lo;

t           = t( wlo:hi );
Mom_cv      = Mom_cv( wlo:hi,: );
Mom_netFlux = Mom_netFlux( wlo:hi,: );


%__________________________________
% Find the time rate of change of the momentum in the control volume (first order backward differenc)

for i = 2:length(t)
    dMdt(i,1:3) = ( Mom_cv(i,:) - Mom_cv(i-1,:) )./( t(i) - t(i-1) );

    if(  ( t(i) - t(i-1) ) == 0 )
      printf(' %s detetected 0, t(i): %e, t(i-1), %e \n', uda{j}, t(i), t(i-1) );
      printf(' This is probably due to overlap in the data when a restart occurs');
    endif

   force(i,1:3) = dMdt(i,:) + Mom_netFlux(i,:);
end

%__________________________________
% Compute a moving average the variables
% Useful if the data is noisy
ave.Mom_netFlux = moveAve( Mom_netFlux, window );
ave.Mom_cv      = moveAve( Mom_cv,      window);
t_crop          = t( window:length(t) );

%__________________________________
% compute force and dM/dt with the averaged quantities
ave.dMdt  = zeros;
ave.force = zeros;

for i = 2:length( ave.Mom_cv )
   ave.dMdt(i,1:3)  = ( ave.Mom_cv(i,:) - ave.Mom_cv(i-1,:) )./( t_crop(i) - t_crop(i-1) );
   ave.force(i,1:3) =   ave.dMdt(i,:) + ave.Mom_netFlux(i,:);
end

movingAveWindow = t(window) - t(1)

printf( 'Loaded uda: %s, maximum time: %e \n',me, max(t))


%______________________________________________________________________
%  Plot the momentum quantities, instantaneous and averaged

%  Momentum Flux (  instaneous )
graphics_toolkit("gnuplot")
h = figure(1);
subplot(2,1,1)
ax = plot(t, Mom_netFlux );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'Net momentum flux' )
xlim( [tmin, tmax] );
grid('on');

%  Average
subplot(2,1,2)

ax = plot( t_crop, ave.Mom_netFlux );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'Average Net momentum flux' )
xlim( [tmin, tmax] );
grid('on');
%hardcopy(h, "Mom_netFlux.jpg", pausePlot);


%__________________________________
%  Control Volume Momentum (instantenous)
%__________________________________

h = figure(2);

subplot( 2,1,1 )
ax = plot( t, Mom_cv );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'Total Momentum in CV' )
xlim( [tmin, tmax] );
grid( 'on' );

%  Average
subplot( 2,1,2 )
ax = plot( t_crop, ave.Mom_cv );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'Average momentum in CV' )
xlim( [tmin, tmax] );
grid('on');

%hardcopy(h, "Mom_netFlux.jpg", pausePlot);

%__________________________________
%  dM/dt   (instantenous)
%__________________________________

h = figure(3);

subplot( 2,1,1 )
ax = plot( t, dMdt );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'dM / dt' )
xlim( [tmin, tmax] );
grid( 'on' );

%  Average
subplot( 2,1,2 )
ax = plot( t_crop, ave.dMdt );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'ave dM / dt' )
xlim( [tmin, tmax] );
grid( 'on' );

%__________________________________
%  Force
%__________________________________

h = figure(3);

subplot( 2,1,1 )
ax = plot( t, force );

legend( "x", "y", "z" )
title( baseTitle );
ylabel( 'Force' )
xlabel( 'Time [s] ');
xlim( [tmin, tmax] );
grid( 'on' );


subplot( 2,1,2 )
ax = plot( t_crop, ave.force );

legend( "x", "y", "z" )
title( baseTitle );
xlabel( 'Time [s] ');
ylabel( 'Average Force' )
xlim( [tmin, tmax] );
grid( 'on' );

pause

exit
