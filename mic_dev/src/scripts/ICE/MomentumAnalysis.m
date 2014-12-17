#!/usr/bin/octave -qf
pkg load signal    %  needed for downsample

%______________________________________________________________________
% This postprocessing script reads in data from the DataAnalysis:momentumAnalysis module
% and calculates the forces on the the system.  

% The file is assumed to have the following format:
% #                                                 total momentum in the control volume                                          Net convective momentum flux                                               net viscous flux                                                             pressure force on control vol.
% #Time                    CV_mom.x                 CV_mom.y                  CV_mom.z                  momFlux.x               momFlux.y                momFlux.z                 visFlux.x                 visFlux.y                visFlux.z                 pressForce.x              pressForce.y             pressForce.z
% 0.000000000000000E+00,   5.000000001498741E-12,   0.000000000000000E+00,   0.000000000000000E+00,   -1.059955609614705E-16,   0.000000000000000E+00,   0.000000000000000E+00,   0.000000000000000E+00,   0.000000000000000E+00,   0.000000000000000E+00,   5.000000001302851E-03,   0.000000000000000E+00,   0.000000000000000E+00
% 1.033217624577908E-02,   1.030545874978994E-04,   -9.182906439727312E-18,   0.000000000000000E+00,   -6.750499239012268E-07,   5.917735138002064E-19,   0.000000000000000E+00,   6.506779409128762E-04,   -2.725043702888364E-15,   0.000000000000000E+00,   9.976081673059412E-03,   0.000000000000000E+00,   0.000000000000000E+00
% 2.066431568137475E-02,   1.975363171601296E-04,   -2.120405362516817E-17,   0.000000000000000E+00,   1.798595651158136E-06,   4.675240092628815E-18,   0.000000000000000E+00,   9.299633795377633E-04,   -1.565569555792368E-14,   0.000000000000000E+00,   9.986773074714961E-03,   -2.273736754432321E-13,   0.000000000000000E+00
% 3.099645988997394E-02,   2.896522155858320E-04,   -9.492698005770271E-19,   0.000000000000000E+00,   -1.074383467203782E-06,   -1.088018360951892E-18,   0.000000000000000E+00,   1.142972503807601E-03,   1.127344720319106E-14,   0.000000000000000E+00,   9.947542064651316E-03,   0.000000000000000E+00,   0.000000000000000E+00
% 4.132860237568741E-02,   3.797903764995445E-04,   -1.046402282171744E-17,   0.000000000000000E+00,   -1.288333479558374E-06,   6.562655332393807E-19,   0.000000000000000E+00,   1.323103564302538E-03,   -1.122311739220283E-14,   0.000000000000000E+00,   9.954688616019780E-03,   4.547473508864641E-13,   0.000000000000000E+00
% 5.166074258976499E-02,   4.681608456181288E-04,   -1.047002793263888E-17,   0.000000000000000E+00,   2.792826493692855E-06,   -6.791419855725228E-18,   0.000000000000000E+00,   1.480952750495167E-03,   7.436454943832528E-15,   0.000000000000000E+00,   9.966911900662012E-03,   2.273736754432321E-13,   0.000000000000000E+00
%
% Below are inputfile specifications:
% 
%     <DataAnalysis>
%       <Module name = "momentumAnalysis">
%         <materialIndex> 0 </materialIndex>       
% 
%         <samplingFrequency> 1e2 </samplingFrequency>
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



%______________________________________________________________________
% This function computes the spectrum and plots the 
% amplitute versus frequency

function [freq, Amp] =  plotFFT( time, Q, desc )
  %http://www.mathworks.com/help/matlab/ref/fft.html
  r    = length( time );
  NFFT = 2^nextpow2(r)          % Next por of 2 from length of r

  % compute the spectrum
  X    = abs( fft( Q, NFFT) );
  Amp  = 2 * abs ( X(1:NFFT/2 + 1) );
  
  % compute a mean dt
  for i = 2:length( time )
    diff = abs( time(i) - time(i-1) );
  end
  dt = mean(diff);

  % create frequency array
  Fs = 1/dt;
  freq = Fs/2 * linspace(0,1, NFFT/2+1);
  
  [maxAmp, index]  = max( Amp );  
  printf(" max Amplitude at %e Hz, max Amplitude: %e\n",freq(index), Amp(index));

  % plot the results
  figure()
  plot(freq, Amp);
  legend ( desc );
  grid   ( 'on');
  title  ( 'Amplititude Spectrum' );
  xlabel ( 'Frequency (Hz) ');
  ylabel ( '|Y(f)|' );
  xlim   ([0 100])
  ylim   ([ 0 1e-4])
  pause
endfunction



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
% clean out duplicate entries

function [cleanData] = removeDupes( data)

t = data(:,1);
count = 1;
for i = 2:length( t )
  if(  ( t(i) - t(i-1) ) != 0 )
    cleanData(count,:) = data(i,:);
    count += 1;
  else
    printf( ' detected duplicate entries at time %e, now removing them.\n', t(i) );
  endif
end
endfunction

%______________________________________________________________________
%  convert string to bool

function [ me ] = str2bool( input)
  me = -9;
  if( strcmp(input, "true") || strcmp( input, "True") || strcmp( input, "TRUE" ) )
    me = 1;
  endif
  
  if( strcmp(input, "false") || strcmp( input, "False") || strcmp( input, "FALSE" ) )
    me = 0;
  endif
  
  if (me == -9)
    printf("ERROR:  ( %s ) is neither true or false, please correct \n", input);
    printf("        now exiting." );
    exit(1)
  endif
endfunction

%______________________________________________________________________
% generate a hardcopy

function hardcopy(h, filename, hardcopy)
  if ( !hardcopy )
    return;
  endif

  FN = findall(h,'-property','FontName');
  set(FN,'FontName','Times');
  
  FS = findall(h, '-property','FontSize');
  set(FS,'FontSize',12);
  
 % PS = findall(h, '-property','papersize');
 % set(PS, 'papersize', [2.5, 3.5])
 set(h, 'paperorientation', 'landscape');
 
 set(h, 'position', [30,30,1024,840]);
 
 saveas( h, filename, 'pdf')

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
  printf( "  -downsample    [1]                Down sampling: include every Nth point in analysis\n" );
  printf( "  -window        [0]                Amount of time used in the average windowing\n" );
  printf( "  -hardCopy     [false]             Produce hard copy of plots\n" );
  printf( "  -createPlots  [true]              Produce plots of: \n" );
  printf( "                                      Net momentum flux vs time       (Instaneous / average )\n");
  printf( "                                      Control volume momentum vs time (Instaneous / average )\n");
  printf( "                                      Force vs time                   (Instaneous / average )\n");
    
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
  if ( opt.tmin < 0 || opt.tmax < 0)
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
opt.help     =  false ;
opt.tmin     =  0.0 ;
opt.tmax     =  1000 ;
opt.uda      = "";
opt.window   = 1e-20;
opt.nPoints  = 1;
opt.hardcopy = false;
opt.title    = "";
opt.datFile  = "notSet";
opt.doPlots  = true ;

% Process command line options
args = argv();
i = 1 ;

if(nargin < 2 )
  usage(argv);
  exit
endif

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
  case { "-hardCopy" }
      opt.hardcopy = str2bool( value );
  case { "-createPlots" }
      opt.doPlots = str2bool( value );
  otherwise
      printf("Unknown input option %s \n\n", option) ;
      usage(argv);
      exit(1)
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


% remove duplicate rows. 
data = removeDupes( data );

%__________________________________
% downsample the data with every nth element
t                     =   data(:,1);      % column 1                time
Mom_cv                =   data(:,2:4);    % columns 2 - 4           control volume momentum
Mom_netFaceFlux       =   data(:,5:7);    % columns 5 - 7           net convective momentum flux across faces 
Viscous_netFaceFlux   =   data(:,8:10);   % columns 8 - 10          net viscous flux across faces
surfacePressForce     =   data(:,11:13);  % columns 11 - 12         net pressure forces on faces

%__________________________________
%  Allow user to trim data between tmin and tmax
opt.tmax    = min( max(t), opt.tmax);

[lo,hi] = trim( t, opt.tmin, opt.tmax );

croppedTime   = [ t(lo)- t(1), t(length(t)) - t(hi)];
croppedPoints = length(t) - hi;
printf( "    - Now removing the leading (%i points, %4.3g sec) and trailing  (%i points, %4.3g sec) from data\n", lo, croppedTime(1), croppedPoints, croppedTime(2)  )

t = t( lo:hi );
Mom_cv              = Mom_cv( lo:hi,: );
Mom_netFaceFlux     = Mom_netFaceFlux( lo:hi,: );
Viscous_netFaceFlux = Viscous_netFaceFlux( lo:hi,: );
surfacePressForce   = surfacePressForce( lo:hi,: );


%__________________________________
% Find the time rate of change of the momentum in the control volume (first order backward differenc)
printf( "    - Now computing intantaneous time rate of change of momentum in the system and the force\n"   );

for i = 2:length(t)
    dMdt(i,1:3) = ( Mom_cv(i,:) - Mom_cv(i-1,:) )./( t(i) - t(i-1) );

    if(  ( t(i) - t(i-1) ) == 0 )
      printf(' %s detetected 0, t(i): %e, t(i-1), %e \n', opt.datFile, t(i), t(i-1) );
      printf(' This is probably due to overlap in the data when a restart occurs');
    endif
    
   force(i,1:3) = dMdt(i,:) + Mom_netFaceFlux(i,:) - Viscous_netFaceFlux(i,:) - surfacePressForce(i,:);
end

%__________________________________
% Compute a moving average the variables
% Useful if the data is noisy
printf( "    - Now computing moving averages of the momentum flux and momentum in control volume\n"   );
[ ave.Mom_netFaceFlux, size ] = moveAve( Mom_netFaceFlux,      opt.window, t );
[ ave.Vis_netFaceFlux, size ] = moveAve( Viscous_netFaceFlux,  opt.window, t );
[ ave.Mom_cv, size ]          = moveAve( Mom_cv,               opt.window, t );
[ ave.surfPressForce, size ]  = moveAve( surfacePressForce,    opt.window, t );


t_crop = t(size);

%__________________________________
% compute force and dM/dt with the averaged quantities
printf( "    - Now computing average quantities\n"   );
ave.dMdt  = zeros;
ave.force = zeros;

for i = 2:length( ave.Mom_cv )
   ave.dMdt(i,1:3)  = ( ave.Mom_cv(i,:) - ave.Mom_cv(i-1,:) )./( t_crop(i) - t_crop(i-1) );
   ave.force(i,1:3) =   ave.dMdt(i,:) + ave.Mom_netFaceFlux(i,:) - ave.Vis_netFaceFlux(i,:) - ave.surfPressForce(i,:);
end

meanForce = mean ( force );
meanV_force = mean ( Viscous_netFaceFlux );
meanP_force = mean ( surfacePressForce );
printf( '______________________________________________________________________\n');
printf( '  Mean force %e, %e, %e \n', meanForce(1), meanForce(2), meanForce(3) );
printf( '\n  Mean forces on the control volume surfaces \n' );
printf( '  viscous:  %e, %e, %e \n', meanV_force(1), meanV_force(2), meanV_force(3) );
printf( '  pressure: %e, %e, %e \n', meanP_force(1), meanP_force(2), meanP_force(3) );
printf( '______________________________________________________________________\n');


%______________________________________________________________________
%  Write to files
%______________________________________________________________________
fp = fopen ("Force.dat", "w");
fprintf( fp, "# time                force.x                force.y                force.z\n");
for i = 1:length( t )
  fprintf( fp, "%15.14e, %15.14e, %15.14e, %15.14e\n", t(i), force( i,1 ), force( i,2 ), force( i,3 ) );
end

fp = fopen ("aveForce.dat", "w");
fprintf( fp, "# time             ave.force.x             aave.force.y             ave.force.z\n");
for i = 1:length( t_crop )
  fprintf( fp, "%15.14e, %15.14e, %15.14e, %15.14e\n", t_crop(i), ave.force( i,1 ), ave.force( i,2 ), ave.force( i,3 ) );
end


%______________________________________________________________________
%  Plot the momentum quantities, instantaneous and averaged
%______________________________________________________________________

if( opt.doPlots )

  %  Momentum Flux (  instaneous )
  graphics_toolkit("gnuplot")
  h = figure(1);
  subplot(2,1,1)
  ax = plot(t, Mom_netFaceFlux );

  legend( "x", "y", "z" )
  title( opt.title );
  xlabel( 'Time [s] ');
  ylabel( 'Net momentum flux' )
  xlim( [opt.tmin, opt.tmax] );
  grid('on');

  %  Average
  subplot(2,1,2)

  ax = plot( t_crop, ave.Mom_netFaceFlux );

  legend( "x", "y", "z" )
  xlabel( 'Time [s] ');
  ylabel( 'Average Net momentum flux' )
  xlim( [opt.tmin, opt.tmax] );
  grid('on');
  hardcopy(h, "momentum_netFlux.pdf", opt.hardcopy );


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
  xlabel( 'Time [s] ');
  ylabel( 'Average momentum in CV' )
  xlim( [opt.tmin, opt.tmax] );
  grid('on');

  hardcopy(h, "momentum_CV.pdf", opt.hardcopy );

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
  xlabel( 'Time [s] ');
  ylabel( 'ave dM / dt' )
  xlim( [opt.tmin, opt.tmax] );
  grid( 'on' );
  
  hardcopy(h, "dmomentum_dt.pdf", opt.hardcopy );


  %__________________________________
  %  viscous
  %__________________________________

  h = figure(4);

  ax = plot( t, Viscous_netFaceFlux );

  legend( "x", "y", "z" )
  title( opt.title );
  xlabel( 'Time [s] ');
  ylabel( 'Net Viscous Surface Force ' )
  xlim( [opt.tmin, opt.tmax] );
  grid( 'on' );

  hardcopy(h, "viscous.pdf", opt.hardcopy );
  
  %__________________________________
  %  pressure forces
  %__________________________________

  h = figure(5);

  ax = plot( t, surfacePressForce );

  legend( "x", "y", "z" )
  title( opt.title );
  xlabel( 'Time [s] ');
  ylabel( 'Net Surface Pressure Force ' )
  xlim( [opt.tmin, opt.tmax] );
  grid( 'on' );

  hardcopy(h, "pressureForce.pdf", opt.hardcopy );


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
  xlabel( 'Time [s] ');
  ylabel( 'Average Force' )
  xlim( [opt.tmin, opt.tmax] );
  grid( 'on' );

  hardcopy(h, "force.pdf", opt.hardcopy );
  pause

endif

%______________________________________________________________________
[freq, amp] = plotFFT( t, force(:,2), "PowerSpectrum of force in Y dir" );



exit
