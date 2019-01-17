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

function Usage
  printf('compare_Couette-Poiseuille.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName>     - name of the uda file \n')  
  printf('  -pDir <0,1,2>           - principal direction \n')           
  printf('  -sliceDir <0,1,2>       - direction for slicing \n')                                                   
  printf('  -mat                    - material index \n')          
  printf('  -wallVel                - wall velocity \n' )                                              
  printf('  -plot <true, false>     - produce a plot \n') 
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
makePlot    = "true";
ts          = 999;
myP         = -9;
output_file = 'L2norm';
L           = 0;

% Parse the command line arguments
arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-uda") )   
    uda      = opt_value;
    
  elseif (strcmp(option,"-pDir") ) 
    pDir     = str2num(opt_value) + 1;
    
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

if ( myP == 1)
  desc    = 'Couette-Poiseuille Flow (P = -0.25, U = 3.4453, dpdx = 100)'
  wallVel = 3.4453  
elseif ( myP == -1 )
  desc    = 'Couette-Poiseuille Flow (P = -0.25, U = 3.4453, dpdx = -100)'
  wallVel = 3.4453
elseif ( myP == -0.25 )
  desc    = 'Couette-Poiseuille Flow (P = -0.25, U = 13.781, dpdx = -100)'
  wallVel = 13.781
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

%__________________________________
% find lineextract limits

lo  = zeros(3,1);
hi  = zeros(3,1);
lo(sliceDir) = -1;
hi(sliceDir) = resolution(sliceDir);
lo(pDir)     = resolution(pDir)/2;
hi(pDir)     = lo(pDir);

startEnd = sprintf('-istart %i %i %i -iend %i %i %i', lo(1), lo(2), lo(3), hi(1), hi(2), hi(3) );

c1 = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o vel.dat -m %i  -uda %s  > /dev/null 2>&1',L, ts-1, startEnd, mat, uda);
[s1, r1] = unix(c1);

%__________________________________
% import the velocity into array
vel     = load('vel.dat'); 
y_CC    = vel(:,sliceDir);
vel_sim = vel(:,3 + pDir);


%__________________________________
% extract dpdx from uda   Need to generalize this
lo  = zeros(3,1);
hi  = zeros(3,1);
lo(pDir) = 0;
hi(pDir) = resolution(pDir);
lo(sliceDir) = resolution(sliceDir)/2;
hi(sliceDir) = lo(sliceDir);

if(pDir == 1)
  p_FC = "pressX_FC"
elseif(pDir == 2)
  p_FC = "pressY_FC"  
elseif(pDir == 3)
  p_FC = "pressZ_FC"           
end

startEnd = sprintf('-istart %i %i %i -iend %i %i %i', lo(1), lo(2), lo(3), hi(1), hi(2), hi(3) );

c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o press_FC.dat -m %i  -uda %s  > /dev/null 2>&1',p_FC, L, ts-1, startEnd, mat, uda);
[s1, r1] = unix(c1);


%__________________________________
% import the press_FC
tmp      = load('press_FC.dat');
x_FC     = tmp(:,pDir);
xLo      = 1;
xHi      = hi(pDir);
press_FC = tmp(:,3 + pDir);
dpdx     = (  press_FC(xLo) - press_FC(xHi) )/( x_FC(xLo) - x_FC(xHi)  )


%______________________________
% computes exact solution
nu = viscosity;
dy = y_CC(2) - y_CC(1)
h  = ( domainLength(sliceDir) + dy )/2.0
% you need to add dy because the velocity BC is set dy/2 from the edge of wall.

% Exact solution for Combined Couette-Poiseuille flow  Equation 3.42
if ( wallVel != 0.0 )
  disp("Couette-Poiseuille flow")
  P               = -dpdx * ( h^2/(2.0 * nu * wallVel) )
  vel_ratio_exact = 0.5 * (1 + y_CC./h) + P*(1 - (y_CC.^2/h.^2));
  vel_exact       = vel_ratio_exact * wallVel;
end

% Exact solution for pure Poiseuille flow, Equation 3.44
if ( wallVel == 0.0 )
  disp("Poiseuille flow")
  umax            = -dpdx * ( h^2/(2.0 * nu) )
  vel_exact       = umax * (1 - y_CC.^2/h.^2);
end

%______________________________
% compute the L2 norm
clear d;
d = 0;
d = abs(vel_sim - vel_exact);
L2_norm = sqrt( sum(d.^2)/length(y_CC) )

% write L2_norm to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',L2_norm);
  fclose(fid);
end

% cleanup
c = sprintf('mv vel.dat %s ', uda);
[s, r] = unix(c);
[s, r] = unix( '/bin/rm press_FC.dat' );

%______________________________
% Plot the results
if (strcmp(makePlot,"true"))
  graphics_toolkit('gnuplot');
  
  h = figure('position',[100,100,1024,768]);
  set (h, "defaultaxesfontname", "Arial");
  set (h, "defaulttextfontname", "Arial");
  
  subplot(2,1,1)
  plot( vel_sim, y_CC, 'b:o', vel_exact, y_CC, 'r:+')
  
  l = legend( 'uda', 'Analytical' );
  set (l, "fontsize", 8)
  
  xlabel('u velocity')
  ylabel('y')
  title( desc )
  grid on;
 
  subplot(2,1,2)
  plot( d, y_CC, 'b:+');
  xlabel('|u - u_{exact}|'); 
  ylabel('y');
  text( 0.5, 0.5, 'this is a test')
  grid on;
  grid on;
 
  pause(5);
  
  %saves the plot to an output file
  c = sprintf( '%i.jpg',resolution(sliceDir) );

  print ( h, c,'-dpng');
  c = sprintf('mv %s %s ', c, uda);
  [s, r] = unix(c);
end
