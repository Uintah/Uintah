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
function [q_ave, size] = moveAve(Q, window, time)

  % compute window size from mean_delt
  % compute a mean dt
  for i = 2:length( time )
    delt = abs( time(i) - time(i-1) );
  end
  mean_delt = mean(delt);

  windowsize = round(window / mean_delt);
  windowsize  = max( 1, windowsize );

  q_tmp(:,1) = filter( ones(windowsize,1)/windowsize, 1, Q(:,1) );  % X-dir
  q_tmp(:,2) = filter( ones(windowsize,1)/windowsize, 1, Q(:,2) );  % Y
  q_tmp(:,3) = filter( ones(windowsize,1)/windowsize, 1, Q(:,3) );  % Z

  % trim off the first "windowsize" data points.  They're junk
  lo = windowsize;
  hi = length( q_tmp(:,1) );
  q_ave = q_tmp( lo:hi, 1:3 );
  size = ( lo:hi );

endfunction


%______________________________________________________________________
% find lower and upper indicies

function [lo, hi] = trim( t, tLo, tHi)

  lo = min( find( t>=tLo ) );
  hi = max( find( t<=tHi ) );
endfunction

%______________________________________________________________________
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
%  Display usage

function usage()
  printf( " MomentumAnalysis  [options]   -file uda/datafile\n\n" );
  printf( "______________________________________________________________________\n");
  printf( " [options]       [default value]    [description] \n\n" );
  
  printf( "  -tmin          [0]                Physical time to start analysis\n" );       
  printf( "  -tmax          [1000]             Physical time to end analysis\n" );
  printf( "  -title         []                 Title for plots\n" );
  printf( "  -downsample    [0]                Down sampling: include every Nth point in analysis\n" );
  printf( "  -window        [0]                Amount of time used in the average opt.windowing\n" );
  printf( "  -hardCopy     [false]             Produce hard copy of plots\n" );      
    
endfunction

%______________________________________________________________________
%  
function inputBulletProofing(opt, argumentlist)
  errorFlag = false;
  
  if ( exist( opt.datFile, "file" ) == 0 )
    printf ( "ERROR: The data file (%s) was not found. \n", opt.datFile );
    errorFlag = true;
  endif
  
  %__________________________________  
  %  Look for negative values
  if ( opt.tmin < 0 | opt.tmax < 0)
    printf ( "ERROR: tmin( %d ) or tmax( %d ) is negative. \n", opt.tmin, opt.tmax );
    errorFlag = true;
  endif
  
  if ( opt.nPoints < 0 )
    printf ( "ERROR: downsample( %d ) is negative. \n", opt.nPoints );
    errorFlag = true; 
  endif
  
  if ( opt.window < 0 )
    printf ( "ERROR: window( %d ) is negative. \n", opt.window );
    errorFlag = true; 
  endif
  
  if (errorFlag)
    printf("\n\n");
    argumentlist
    usage()
    printf ( "\n Now exiting..... \n");
    exit(1)
  endif 
   
endfunction

%______________________________________________________________________
%   USER INPUTS
%______________________________________________________________________

% Command line defaults
opt.help    =  false ;
opt.tmin    =  0.0 ;
opt.tmax    =  1000 ;
opt.uda     = "";
opt.window  = 1e-20;
opt.nPoints = 1;
opt.harcopy = false;
opt.title   = "";
opt.datFile = "notSet";

% Process command line options
args = argv();
i = 1 ;
while i <= length(args)
  option = args{i};
  value  = args{++i};
 
  switch option
  case { "-h" "--help" }
      opt.help   = true;
  case { "-tmin" }
      opt.tmin   = str2num( value );
  case { "-tmax" }
      opt.tmax   = str2num( value );
  case { "-window" }
      opt.window = str2num( value );
  case { "-title" }
      opt.title  = value;
  case { "-downSample" }
      opt.nPoints = str2num( value);
  case { "-file" }
      opt.datFile = value;
  otherwise
      break ;
  endswitch
  i++ ;
endwhile

inputBulletProofing(opt, args);

if (opt.help)
  opt
  usage(argv) ;
  exit(1) ;
endif

pausePlot = "true";

format long e



%______________________________________________________________________
%     MAIN  
%______________________________________________________________________

%  Load the data into arrays and compute dM/dt where M is the total momentum
%  in the control volume
data = dlmread( opt.datFile, ",", 4,0 );

%__________________________________
% downsample the data with every nth element
t           = downsample( data(:,1),   opt.nPoints );  % column 1
Mom_cv      = downsample( data(:,2:4), opt.nPoints );  % columns 2 - 4
Mom_netFlux = downsample( data(:,5:7), opt.nPoints );  % columns 5 - 7

%__________________________________
%  Allow user to trim data between tmin and tmax
opt.tmax    = min( max(t), opt.tmax)

[lo,hi] = trim( t, opt.tmin, opt.tmax )

t           = t( lo:hi );
Mom_cv      = Mom_cv( lo:hi,: );
Mom_netFlux = Mom_netFlux( lo:hi,: );


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

[ ave.Mom_netFlux, size ] = moveAve( Mom_netFlux, opt.window, t );
[ ave.Mom_cv, size ]      = moveAve( Mom_cv,      opt.window, t );

t_crop = t(size);

%__________________________________
% compute force and dM/dt with the averaged quantities
ave.dMdt  = zeros;
ave.force = zeros;

for i = 2:length( ave.Mom_cv )
   ave.dMdt(i,1:3)  = ( ave.Mom_cv(i,:) - ave.Mom_cv(i-1,:) )./( t_crop(i) - t_crop(i-1) );
   ave.force(i,1:3) =   ave.dMdt(i,:) + ave.Mom_netFlux(i,:);
end


printf( 'Loaded file: %s, maximum time: %e \n',opt.datFile, max(t))


%______________________________________________________________________
%  Plot the momentum quantities, instantaneous and averaged

%  Momentum Flux (  instaneous )
graphics_toolkit("gnuplot")
h = figure(1);
subplot(2,1,1)
ax = plot(t, Mom_netFlux );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'Net momentum flux' )
xlim( [opt.tmin, opt.tmax] );
grid('on');

%  Average
subplot(2,1,2)

ax = plot( t_crop, ave.Mom_netFlux );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'Average Net momentum flux' )
xlim( [opt.tmin, opt.tmax] );
grid('on');
%hardcopy(h, "Mom_netFlux.jpg", pausePlot);


%__________________________________
%  Control Volume Momentum (instantenous)
%__________________________________

h = figure(2);

subplot( 2,1,1 )
ax = plot( t, Mom_cv );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'Total Momentum in CV' )
xlim( [opt.tmin, opt.tmax] );
grid( 'on' );

%  Average
subplot( 2,1,2 )
ax = plot( t_crop, ave.Mom_cv );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'Average momentum in CV' )
xlim( [opt.tmin, opt.tmax] );
grid('on');

%hardcopy(h, "Mom_netFlux.jpg", pausePlot);

%__________________________________
%  dM/dt   (instantenous)
%__________________________________

h = figure(3);

subplot( 2,1,1 )
ax = plot( t, dMdt );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'dM / dt' )
xlim( [opt.tmin, opt.tmax] );
grid( 'on' );

%  Average
subplot( 2,1,2 )
ax = plot( t_crop, ave.dMdt );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'ave dM / dt' )
xlim( [opt.tmin, opt.tmax] );
grid( 'on' );

%__________________________________
%  Force
%__________________________________

h = figure(3);

subplot( 2,1,1 )
ax = plot( t, force );

legend( "x", "y", "z" )
title( opt.title );
ylabel( 'Force' )
xlabel( 'Time [s] ');
xlim( [opt.tmin, opt.tmax] );
grid( 'on' );


subplot( 2,1,2 )
ax = plot( t_crop, ave.force );

legend( "x", "y", "z" )
title( opt.title );
xlabel( 'Time [s] ');
ylabel( 'Average Force' )
xlim( [opt.tmin, opt.tmax] );
grid( 'on' );

pause

exit
