#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the combined couette-poiseuille flow between plates
% with and without dimensions
% Reference:  Viscous Fluid Flow, 2nd edition, Frank White, pg 118
%
%  Example usage:
%  compare_Couette-Poiseuille.m -sliceDir 1 -mat 0 -plot false -o out.400.cmp -uda rayleigh_400.uda
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
  printf('  -ts                     - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>              - Dump the output (L2Error) to a file\n')    
  printf('----------------------------------------------------------\n')
  printf(' You must modify the hardcoded variables\n')
  printf(' wallVel:  \t wall velocity \n viscosity \n dpdx:  \t pressure gradient\n')     
  printf('----------------------------------------------------------\n')                         
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
% add function directory to search path
myPath   = which( mfilename );
srcPath  = readlink( myPath );
funcPath = strcat( fileparts (srcPath), "/functions" );
addpath( funcPath )

%__________________________________
% USER DEFINED VARIABLE
wallVel        = 0.0
viscosity      = 1e-3
dpdx           = -(101325 - 101315)/(1.0)

%__________________________________
% defaults
symbol   = {'+','*r','xg'}; 
pDir        = 1;
sliceDir    = 2;
mat         = 0;
makePlot    = "true";
ts          = 999;
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
                      
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%__________________________________
% extract time and grid info on this level
tg_info = getTimeGridInfo( uda, ts, L );

ts           = tg_info.ts;
resolution   = tg_info.resolution;
domainLength = tg_info.domainLength;


%__________________________________
% find lineextract limits

lo  = zeros(3,1);
hi  = zeros(3,1);
lo(sliceDir) = -1;
hi(sliceDir) = resolution(sliceDir);
lo(pDir)     = resolution(pDir)/2;
hi(pDir)     = lo(pDir);

startEnd = sprintf('-istart %i %i %i -iend %i %i %i', lo(1), lo(2), lo(3), hi(1), hi(2), hi(3) );

c1 = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o vel.dat -m %i  -uda %s  > /dev/null 2>&1',L, ts, startEnd, mat, uda);
[s1, r1] = unix(c1);

%__________________________________
% import the velocity into array
vel     = load('vel.dat'); 
y_CC    = vel(:,sliceDir);
vel_sim = vel(:,3 + pDir);

%______________________________
% computes exact solution
nu = viscosity;
dy = y_CC(2) - y_CC(1)
h  = (domainLength(sliceDir) + dy)/2.0

% you need to add dy because the velocity BC is set dy/2 from the edge of wall.

% Exact solution for Combined Couette-Poiseuille flow
if (wallVel > 0.0)
  disp("Couette-Poiseuille flow")
  P               = -dpdx * ( h^2/(2.0 * nu * wallVel) )
  vel_ratio_exact = 0.5 * (1 + y_CC./h) + P*(1 - (y_CC.^2/h.^2));
  vel_exact       = vel_ratio_exact * wallVel;
end

% Exact solution for pure Poiseuille flow
if (wallVel == 0.0 )
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

%______________________________
% Plot the results
if (strcmp(makePlot,"true"))
  hf = figure();
%  set (hf, "visible", "off");
  
  subplot(2,1,1)
  plot( vel_sim, y_CC, 'b:o;computed;', vel_exact, y_CC, 'r:+;exact;')
  xlabel('u velocity')
  ylabel('y')
  title('Combined Couette-Poiseuille Flow');
  grid on;
%  hold on;
  
  subplot(2,1,2)
  plot( d, y_CC, 'b:+');
  xlabel('|u - u_{exact}|'); 
  ylabel('y');
  grid on;
  hold off;
  
% set (hf, "visible", "on");
  pause(7)
  
  unix('/bin/rm Couette-Poiseuille.png > /dev/null 2>&1');
  print( hf, 'Couette-Poiseuille.png','-dpng', '-FTimes-Roman:14');
  c = sprintf('mv Couette-Poiseuille.png %s ', uda);
  [s, r] = unix(c);
end
