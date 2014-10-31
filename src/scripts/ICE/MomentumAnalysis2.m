#!/usr/bin/octave -qf

%______________________________________________________________________
% This postprocessing script reads in a data file that was computed using
%
%     puda -ICE_momentum  -mat <X>  < uda >
%
% and calculates the forces on the the system.  

% The file is assumed to have the following format:
% #                                                 total momentum in the control volume                                          Net convective momentum flux                                                                net viscous flux                                                         pressure force on control vol.
% #Time                      CV_mom.x                   CV_mom.y                    CV_mom.z                    momFlux.x                  momFlux.y                    momFlux.z                    visFlux.x                  visFlux.y                   visFlux.z                  pressForce.x               pressForce.y                pressForce.z               mDot.x                      mDot.y                      mDot.z
% 1.015677475043015E-01      1.518767761434530E-04      -2.497897223790831E-16       0.000000000000000E+00       2.322523683789823E-15      7.775377283632840E-17       0.000000000000000E+00       -7.764866510202561E-04       9.010514909516351E-15       0.000000000000000E+00      -1.421085471520200E-14      0.000000000000000E+00       0.000000000000000E+00      -1.196146046794633E-16      0.000000000000000E+00       0.000000000000000E+00
% 2.006326145560243E-01      2.152010948034806E-04      -5.422664833526491E-16       0.000000000000000E+00       3.924323973419908E-15      -3.367510048791324E-16       0.000000000000000E+00       -5.463527608391332E-04       9.451389358658936E-15       0.000000000000000E+00      -1.421085471520200E-14      0.000000000000000E+00       0.000000000000000E+00      -1.400789206851272E-16      0.000000000000000E+00       0.000000000000000E+00
% 3.017176572171121E-01      2.646243596651752E-04      -1.635839374535563E-16       0.000000000000000E+00       -3.539920093165172E-15      8.486457404891147E-16       0.000000000000000E+00       -4.438702775214192E-04       2.738435050603368E-15       0.000000000000000E+00      0.000000000000000E+00      0.000000000000000E+00       0.000000000000000E+00      -1.547698601223058E-16      0.000000000000000E+00       0.000000000000000E+00
%  
%     Input file must have the following labels saved.
%       <save label="rho_CC"/>
%       <save label="temp_CC"/>
%       
%       <save label = "pressX_FC"/>
%       <save label = "pressY_FC"/>
%       <save label = "pressZ_FC"/>
%       
%       <save label = "uvel_FCME"/>
%       <save label = "vvel_FCME"/>
%       <save label = "wvel_FCME"/>      
%       
%       <save label = "tau_X_FC"/>
%       <save label = "tau_Y_FC"/>
%       <save label = "tau_Z_FC"/>
%______________________________________________________________________


clear all; clc;
close all;

%______________________________________________________________________
%                HELPER FUNCTIONS
%______________________________________________________________________

%______________________________________________________________________
% find lower and upper indicies

function [lo, hi] = trim( t, tLo, tHi)

  lo = min( find( t>=tLo ) );
  hi = max( find( t<=tHi ) );
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
  printf( " MomentumAnalysis  [options]   -file MomentumAnalysis.txt\n\n" );
  printf( "______________________________________________________________________\n");
  printf( " [options]       [default value]    [description] \n\n" );
  
  printf( "  -tmin          [0]                Physical time to start analysis\n" );       
  printf( "  -tmax          [1000]             Physical time to end analysis\n" );
  printf( "  -title         []                 Title for plots\n" );
  printf( "  -hardCopy     [false]             Produce hard copy of plots\n" );
  printf( "  -createPlots  [true]              Produce plots of: \n" );
  printf( "                                      Net convective momentum flux across control surfaces vs time\n");
  printf( "                                      Net viscous momentum flux across control surfaces vs time\n");
  printf( "                                      Control volume momentum vs time \n");
    
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
  case { "-title" }
      opt.title  = value;
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
beginRow = 2;
beginCol = 0;
data = dlmread( opt.datFile, ",", beginRow, beginCol );

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

t           = t( lo:hi );
Mom_cv      = Mom_cv( lo:hi,: );
Mom_netFaceFlux = Mom_netFaceFlux( lo:hi,: );

%__________________________________
% Find the time rate of change of the momentum in the control volume (first order backward differenc)
printf( "    - Now computing intantaneous time rate of change of momentum in the system and the force\n"   );

for i = 2:length(t)
    dMdt(i,1:3) = ( Mom_cv(i,:) - Mom_cv(i-1,:) )./( t(i) - t(i-1) );

    if(  ( t(i) - t(i-1) ) == 0 )
      printf(' %s detetected 0, t(i): %e, t(i-1), %e \n', uda{j}, t(i), t(i-1) );
      printf(' This is probably due to overlap in the data when a restart occurs');
    endif

   force(i,1:3) = dMdt(i,:) + Mom_netFaceFlux(i,:) + Viscous_netFaceFlux(i,:) + surfacePressForce(i,:);
end

meanForce = mean ( force );
printf( '______________________________________________________________________\n');
printf( '  Mean force %e, %e, %e \n', meanForce(1), meanForce(2), meanForce(3) );
printf( '______________________________________________________________________\n');


%______________________________________________________________________
%  Write to files
%______________________________________________________________________
fp = fopen ("Force.dat", "w");
fprintf( fp, "# time                force.x                force.y                force.z\n");
for i = 1:length( t )
  fprintf( fp, "%15.14e, %15.14e, %15.14e, %15.14e\n", t(i), force( i,1 ), force( i,2 ), force( i,3 ) );
end

%______________________________________________________________________
%  Plot the momentum quantities, instantaneous and averaged
%______________________________________________________________________

if( opt.doPlots )

  %  Momentum Flux (  instaneous )
  graphics_toolkit("gnuplot")
  h = figure(1);

  ax = plot(t, Mom_netFaceFlux ); 

  legend( "x", "y", "z" )
  title( opt.title );
  xlabel( 'Time [s] ');
  ylabel( 'Net momentum flux' )
  xlim( [opt.tmin, opt.tmax] );
  grid('on');

  hardcopy(h, "momentum_netFlux.pdf", opt.hardcopy );


  %__________________________________
  %  Control Volume Momentum
  %__________________________________

  h = figure(2);

  ax = plot( t, Mom_cv );

  legend( "x", "y", "z" )
  title( opt.title );
  xlabel( 'Time [s] ');
  ylabel( 'Total Momentum in CV' )
  xlim( [opt.tmin, opt.tmax] );
  grid( 'on' );

  hardcopy(h, "momentum_CV.pdf", opt.hardcopy );

  %__________________________________
  %  dM/dt
  %__________________________________

  h = figure(3);

  ax = plot( t, dMdt );

  legend( "x", "y", "z" )
  title( opt.title );
  xlabel( 'Time [s] ');
  ylabel( 'dM / dt' )
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

  h = figure(6);

  ax = plot( t, force );

  legend( "x", "y", "z" )
  title( opt.title );
  ylabel( 'Force' )
  xlabel( 'Time [s] ');
  xlim( [opt.tmin, opt.tmax] );
  grid( 'on' );

  hardcopy(h, "force.pdf", opt.hardcopy );
  pause

endif

exit
