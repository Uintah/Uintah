#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the rayleigh problem with and without dimensions
% Reference:  Incompressible Flow, by Panton pg 177
%
%  Example usage:
%  compare_Rayleigh.m -aDir 1 -mat 0 -plot false -o out.400.cmp -uda rayleigh_400.uda
%_________________________________

clear all;
close all;
format short e;

function Usage
  printf('compare_Rayleigh.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                 
  printf('  -aDir <1,2,3>       - axial direction \n')                                                   
  printf('  -mat                - material index \n')                                                        
  printf('  -plot <true, false> - produce a plot \n')                                                        
  printf('  -ts                 - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>          - Dump the output (L2Error) to a file\n')                                    
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
% default user inputs
pDir        = 999;
mat         = 0;
makePlot    = true;
ts          = 999;
output_file = 'L2norm';
L           = 0;

arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-uda") )   
    uda = opt_value;
  elseif (strcmp(option,"-aDir") ) 
    pDir = str2num(opt_value);
  elseif (strcmp(option,"-mat") )
    mat = str2num(opt_value);
  elseif (strcmp(option,"-plot") )
    makePlot = opt_value; 
  elseif (strcmp(option,"-ts") )
    ts = str2num(opt_value);                  
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%__________________________________
%  HARDWIRED CONSTANTS
viscosity      = 1e-4;
vel_CC_initial = 10.0;
rho_CC         = 1.1792946927374306;

%__________________________________
% extract time and grid info on this level
tg_info = getTimeGridInfo( uda, ts, L );

ts           = tg_info.ts;
resolution   = tg_info.resolution;
domainLength = tg_info.domainLength;
time         = tg_info.physicalTime;

% find the x and y directions
% for each axial direction different y directions are possible
% these scripts were built for each of the following planes
% when creating new tests you may want to try to follow this convention

if(pDir == 1)
  yDir = 2;
  xDir = 1;
elseif(pDir == 2)
  yDir = 3;
  xDir = 2;
elseif(pDir == 3)
  yDir = 1;
  xDir = 3;
end

%__________________________________
% compute the exact solution & L2Norm

xHalf = resolution(xDir)/2.0;

%sets the start and end cells for the line extract, depending axial direction
if(pDir == 1)
  startEnd = sprintf('-istart %i 0 0 -iend %i %i 0',xHalf,xHalf, resolution(yDir)-1 );
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 %i 0 -iend 0 %i %i',xHalf,xHalf, resolution(yDir)-1 );
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 %i -iend %i 0 %i',xHalf, resolution(yDir)-1 ,xHalf);
end

c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o vel.dat -m %i  -uda %s','vel_CC > /dev/null 2>&1',L,ts-1,startEnd,mat,uda);
[s1, r1] = unix(c1);

%__________________________________
% import the data into arrays
vel  = load('vel.dat'); 
y_CC = vel(:,yDir);

uvel = vel(:,3 + xDir);

%__________________________________
% computes exact solution
vel_ratio_sim = uvel/vel_CC_initial;
nu = viscosity/rho_CC;

vel_ratio_exact =( 1.0 - erf( y_CC/(2.0 * sqrt(nu * time)) ) );
vel_exact = vel_ratio_exact * vel_CC_initial;

%__________________________________
% compute L2norm
clear d;
d = 0;
d = abs(vel_ratio_sim - vel_ratio_exact);
L2_norm = sqrt( sum(d.^2)/length(y_CC) );

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

%______________________________
% Plot the results 
if (strcmp(makePlot,"true"))
  graphics_toolkit('gnuplot')

  h = figure();
  set (h, "defaultaxesfontname", "Arial") 
  set (h, "defaulttextfontname", "Arial")
  
  subplot(2,1,1),
  plot(uvel, y_CC, 'b:o', vel_exact, y_CC, 'r:+')
  
  l = legend( 'uda', 'Analytical' );
  set (l, "fontsize", 8)
  
  xlabel('u velocity')
  ylabel('y')
  title('Rayleigh Problem');
  grid on;

  subplot(2,1,2),
  plot(d, y_CC, 'b:+');
  xlabel('|u - u_{exact}|'); 
  ylabel('y');
  grid on;
  pause(3);   

  %saves the plot to an output file
  c1 = sprintf( '%i.jpg',resolution(yDir) );

  print ( h, c1,'-dpng');
  c = sprintf('mv %s %s ', c1, uda);
  [s, r] = unix(c);
end
