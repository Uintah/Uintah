#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the combined couette-poiseuille flow between plates
% with and without dimensions
% Reference:  Viscous Fluid Flow, 2nd edition, Frank White, pg 118
%
%  Example usage:
%  compare_Couette-Poiseuille.m -sliceDir 1 -mat 0 -P -1 -plot false -o out.400.cmp -uda rayleigh_400.uda
%_________________________________
clear all;
close all;
format short e;

%______________________________________________________________________
% shift from uintah cell indices to octave 1 based arrays
function [O_lo, O_hi] = octaveIndexShift( U_lo, U_hi, includeEC )

  octave_shift = ones(1,3);
  % octave lo and hi
  if( includeEC )
    octave_shift = 2 .* octave_shift;
  end

  O_lo = U_lo + octave_shift;
  O_hi = U_hi + octave_shift;
 
  O_lo = cast( O_lo, "int8" );
  O_hi = cast( O_hi, "int8" );
end


%______________________________________________________________________
% find the low and high indices 
% Uintah:  U_<*>
% octave:  O_<*>     
function [beginEnd] = loHi( axial_transverse, sliceDir, pDir, resolution, includeEC )
 
  % Uintah lo and hi
  
  U_lo = zeros(1,3);
  U_hi = zeros(1,3);
  
  EC = 0;
  if( includeEC )
    EC = 1;
    resolution = resolution;
  else
    resolution = resolution - ones(1,3);
  end  
  
  if( strcmp( axial_transverse,'axial' ) )
    U_lo(pDir)  = -EC;
    U_hi(pDir)  =  U_lo(pDir) + resolution(pDir) + EC;   % inclusive
  
    U_lo(sliceDir) = resolution(sliceDir)/2;
    U_hi(sliceDir) = U_lo(sliceDir);

  elseif ( strcmp( axial_transverse, 'transverse' ) )

    U_lo(sliceDir)  = -EC;
    U_hi(sliceDir)  =  U_lo(sliceDir) + resolution(sliceDir) + EC;   % inclusive
    
    U_lo(pDir)     = resolution(pDir)/2;
    U_hi(pDir)     = U_lo(pDir);
  else
    display( '\n\n***  ERROR: loHi(), axial_transverse is not valid\n\n' );
    exit
  end
  
  [O_lo, O_hi] = octaveIndexShift( U_lo, U_hi, includeEC );
  
  beginEnd.O_lo = O_lo;
  beginEnd.O_hi = O_hi;
  
  beginEnd.U_lo = cast ( U_lo, "int8" );
  beginEnd.U_hi = cast ( U_hi, "int8" );

end

%______________________________________________________________________
% extract variable from uda
function [var, loc, O_lo, O_hi] = extractVar( varName, axial_transverse, sliceDir, pDir, resolution, L, ts, mat, uda, includeEC )
  
  %__________________________________
  % find the high, low indicies
  if( strcmp( axial_transverse,'axial' ) )
    [beginEnd] = loHi( 'axial', sliceDir, pDir, resolution, includeEC );

  elseif ( strcmp( axial_transverse, 'transverse' ) )
    [beginEnd] = loHi( 'transverse', sliceDir, pDir, resolution, includeEC );
  
  else
    display( "\n\n***  ERROR: extractVar(), axial_transverse is not valid\n\n" );
    exit
  end

  U_lo = beginEnd.U_lo;   % uintah cell indices
  U_hi = beginEnd.U_hi;
  O_lo = beginEnd.O_lo;   % corresponding octave indices
  O_hi = beginEnd.O_hi;
  
  startEnd = sprintf('-istart %i %i %i -iend %i %i %i', U_lo(1), U_lo(2), U_lo(3), U_hi(1), U_hi(2), U_hi(3) );

  c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o var.dat -m %i  -uda %s  > /dev/null 2>&1',varName, L, ts-1, startEnd, mat, uda)
  [s, r] = unix(c1);

  %__________________________________
  % import the variable into array
  varTmp  = load('var.dat');
  
  %__________________________________
  % physical location
  if( strcmp( axial_transverse,'axial' ) )
    loc = varTmp(:,pDir);

  elseif ( strcmp( axial_transverse, 'transverse' ) )
    loc = varTmp(:,sliceDir);
  end
  
  %__________________________________
  % variable  
  % is Var double or Vector?
  c = sprintf(" puda -listvariables %s | grep %s | awk -v FS=\"(<|>)\" \'{print $2}\' ", uda, varName );
  [s, dblVec ] = unix(c);
  dblVec = dblVec(1:end-1);  % strip newline
  
  
  if( strcmp( dblVec,'Vector' ) )
    var = varTmp(:,3 + pDir);
    
  elseif( strcmp( dblVec,'double' ) )
    var = varTmp(:,4);
    
  else
    display( '\n\n***  ERROR: extractVar(), dblVec is not valid\n\n' );
    exit
  end
  
  % cleanup
  [s, r] = unix( '/bin/rm var.dat' );
end

%______________________________________________________________________
%______________________________________________________________________
function Usage
  printf('compare_Couette-Poiseuille.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName>     - name of the uda file \n')  
  printf('  -pDir <0,1,2>           - principal direction \n')           
  printf('  -sliceDir <0,1,2>       - direction for slicing \n')                                                   
  printf('  -mat                    - material index \n')          
  printf('  -wallVel                - wall velocity \n' )                                              
  printf('  -plot <true, false>     - produce a plot \n')
  printf('  -periodicBCs            - periodic BCs in principal direction \n')
  printf('  -P <-1, 1, -0.25>       - dimensionless pressure gradient\n')                                              
  printf('  -ts                     - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>              - Dump the output (L2Error) to a file\n')                    
end 

%________________________________            
% Parse User inputs  
%echo
nargin = length(argv);
if (nargin == 0)
  Usage
  exit
endif

%__________________________________
% update paths
myPath   = readlink( mfilename ("fullpathext") );
uintah   = textread( 'scriptPath', '%s','endofline', '\n' );
path     = sprintf( '%s:%s/functions', uintah{:}, fileparts(myPath) );
addpath( path )

% path used by octave:unix command
unixPath = sprintf( '%s:%s', EXEC_PATH(),  uintah{:} );
EXEC_PATH ( unixPath ) 


%__________________________________
% defaults
pDir        = 1;
sliceDir    = 2;
mat         = 0;
L           = 0;
makePlot    = "true";
plotVel     = "true";
plotTau     = "true";
periodicBCs = "false";

ts          = 999;
myP         = -9;
output_file = 'L2norm';

%__________________________________
% Parse the command line arguments

arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-uda") )   
    uda      = opt_value;
    
  elseif (strcmp(option,"-pDir") ) 
    pDir     = str2num(opt_value) + 1;

  elseif (strcmp(option,"-periodicBCs") ) 
    periodicBCs = opt_value;
    
  elseif (strcmp(option,"-sliceDir") ) 
    sliceDir = str2num(opt_value) + 1;
    
  elseif (strcmp(option,"-mat") )
    mat      = str2num(opt_value);
 
  elseif (strcmp(option,"-wallVel") )
    wallVel  = str2num(opt_value);
        
  elseif (strcmp(option,"-plot") )
    makePlot = opt_value;
     
  elseif (strcmp(option,"-ts") )
    ts       = str2num(opt_value);
    
  elseif (strcmp(option,"-P") )
    myP      = str2num(opt_value);                
  
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%__________________________________
% Problem specific variables

viscosity = 1e-2
runID = 0

if ( myP == 1)
  desc    = "Couette-Poiseuille Flow \n(P = 1, U = 3.4453, dpdx = 100 ) \n h = (Ymax - Ymin )/2 ";
  wallVel = 3.4453 
  dpdx    = -100 
elseif ( myP == -1 )
  desc    = "Couette-Poiseuille Flow \n(P = -1, U = 3.4453, dpdx = -100, X +/- BC: periodic) \n h = (Ymax - Ymin )/2 ";
  wallVel = 3.4453
  dpdx    = 100
elseif ( myP == -0.25 )
  desc    = "Couette-Poiseuille Flow \n(P = -0.25, U = 13.781, dpdx = -100, X +/- BC: Periodic) \n h = (Ymax - Ymin)/2 ";
  wallVel = 13.781
  dpdx    = 100
else
  disp( '\n\n***  ERROR: -P has not been specified\n\n' ) 
  Usage
  exit
end

%__________________________________
% extract time and grid info on this level
tg_info = getTimeGridInfo( uda, ts, L );

ts           = tg_info.ts;
resolution   = tg_info.resolution;
domainLength = tg_info.domainLength;
dx           = tg_info.dx;
time         = tg_info.physicalTime
nu           = viscosity;

%__________________________________
% extract dpdx from uda 
periodicBCs
if( strcmp(periodicBCs,"false")  )

  if(pDir == 1)
    p_FC = "pressX_FC"
  elseif(pDir == 2)
    p_FC = "pressY_FC"  
  elseif(pDir == 3)
    p_FC = "pressZ_FC"           
  end

  [press_FC, x_FC, lo, hi] = extractVar( p_FC, 'axial', sliceDir, pDir, resolution, L, ts, mat, uda, true);

  xLo   =  lo(pDir) +1;
  xHi   =  hi(pDir);

  dpdx  = (  press_FC(xLo) - press_FC(xHi) )/( x_FC(xLo) - x_FC(xHi)  )

  printf( 'press_FC(xLo): %f press_FC(xHi): %f', press_FC(xLo), press_FC(xHi) );
  printf( 'x_FC(xLo): %f x_FC(xHi): %f     dpdx: %g\n', x_FC(xLo), x_FC(xHi), dpdx );  
end

%______________________________
%  Simulation Velocity
[vel_sim, y_CC, lo, hi] = extractVar( 'vel_CC', 'transverse', sliceDir, pDir, resolution, L, ts, mat, uda, false);


%__________________________________
% Exact solution for Combined Couette-Poiseuille flow  Equation 3.42

dy   = y_CC(2) - y_CC(1)
h   = ( domainLength(sliceDir) )/2.0

if ( wallVel != 0.0 )
  disp("Couette-Poiseuille flow")
  P               = -dpdx * ( h^2/(2.0 * nu * wallVel) )
  vel_ratio_exact = 0.5 .* (1 .+ y_CC./h) .+ P .* (1 .- (y_CC.^2/h.^2));
  vel_exact       = vel_ratio_exact .* wallVel;
end

% Exact solution for pure Poiseuille flow, Equation 3.44
if ( wallVel == 0.0 )
  disp("Poiseuille flow")
  P               = 0;
  umax            = -dpdx .* ( h^2/(2.0 * nu) )
  vel_exact       = umax .* (1 .- y_CC.^2/h.^2);
end


%______________________________
% compute the L2 norm

clear d;
d = 0;
d = abs(vel_sim .- vel_exact);
L2_norm = sqrt( sum(d.^2)/length(y_CC) )

% write L2_norm to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',L2_norm);
  fclose(fid);
end

%__________________________________
% debugging

printf("       y_CC              vel_sim              vel_exact       diff_vel\n")
for i = 1:length(y_CC)
  printf('%16.15f, %16.15f,  %16.15f,  %16.15g\n',y_CC(i), vel_sim(i), vel_exact(i), d(i) )
endfor


%__________________________________
%  Extract the shear stress and compute the exact solution

if( sliceDir == 1 )
  tau_FC = "tau_X_FC";
elseif( sliceDir == 2 )
  tau_FC = "tau_Y_FC"; 
elseif( sliceDir == 3 )
  tau_FC = "tau_Z_FC";          
end

[tau, y_FC, lo, hi] = extractVar( tau_FC, 'transverse', sliceDir, pDir, resolution, L, ts, mat, uda, true );

tau_sim  = tau( 2: length(tau)  );    % ignore values of tau in the extracells
y_FC     = y_FC(2: length(y_FC) );

dudy      = ( wallVel ./ (2 .* h) ) .- 2 .* P .* wallVel .* y_FC ./ (h .^ 2);
tau_exact = viscosity .* dudy;

diff_tau  = abs( tau_sim .- tau_exact );

printf("       y_FC              tau_sim              tau_exact       diff_tau\n")
for i = 1:length(y_FC)
  printf('%16.15f, %16.15f,  %16.15f,  %16.15g\n',y_FC(i), tau_sim(i), tau_exact(i), diff_tau(i) )
endfor


%______________________________________________________________________
% Plot the results
if (strcmp(makePlot,"true"))
  graphics_toolkit('gnuplot');
  
   %  velocity
  if( strcmp(plotVel, "true" ) )
    h = figure('position',[100,100,1024,768]);
    set (h, "defaultaxesfontname", "Arial");
    set (h, "defaulttextfontname", "Arial");
    set (h,'paperposition', [0,0,[6 7]])
   
    subplot(2,1,1)
    plot( vel_sim, y_CC, 'b:o', vel_exact, y_CC, 'r:+' )

    l = legend( 'uda', 'Analytical' );
    set (l, "fontsize", 8)

    xlabel('u velocity')
    ylabel('y')
    title( desc )
    grid on;

    %  velocity error
    subplot(2,1,2)
    plot( d, y_CC, 'b:+');
    xlabel('|u - u_{exact}|'); 
    ylabel('y');
    grid on;
    pause(3);

    c = sprintf( '%i_vel.%i.png',resolution(sliceDir), runID );
    print ( h, c,'-dpng' );
    c = sprintf('mv %s %s ', c, uda);
    [s, r] = unix(c);
  end

  % shear stress  
  if( strcmp(plotTau, "true" ) )

    h = figure('position',[100,100,1024,768]);
    set (h, "defaultaxesfontname", "Arial");
    set (h, "defaulttextfontname", "Arial");
    set (h,'paperposition', [0,0,[6 7]])
    
    %  velocity
    subplot(2,1,1)
    plot( tau_sim, y_FC, 'b:o', tau_exact, y_FC, 'r:+' )

    l = legend( 'uda', 'Analytical' );
    set (l, "fontsize", 8)

    xlabel('Tau')
    ylabel('y')
    title( desc )
    grid on;

    %  velocity error
    subplot(2,1,2)
    plot( diff_tau, y_FC, 'b:+');
    xlabel('|tau - tau_{exact}|'); 
    ylabel('y');
    grid on;
    pause(3);

    c = sprintf( '%i_tau.%i.png',resolution(sliceDir), runID );
    print ( h, c,'-dpng' );
    c = sprintf('mv %s %s ', c, uda);
    [s, r] = unix(c);
  end 
  
 
  
  
  %saves the plot to an output file

end
