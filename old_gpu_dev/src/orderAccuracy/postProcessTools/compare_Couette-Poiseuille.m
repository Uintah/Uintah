#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the combined couette-poiseuille flow between plates
% with and without dimensions
% Reference:  Viscous Fluid Flow, 2nd edition, Frank White, pg 118
%
%  Example usage:
%  compare_Couette-Poiseuille.m -pDir 1 -mat 0 -plot false -o out.400.cmp -uda rayleigh_400.uda
%_________________________________
clear all;
close all;
format short e;

function Usage
  printf('compare_Couette-Poiseuille.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                  
  %printf('  -pDir <1,2,3>       - principal direction \n')                                                   
  printf('  -mat                - material index \n')                                                        
  printf('  -plot <true, false> - produce a plot \n')                                                        
  printf('  -ts                 - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>          - Dump the output (L2Error) to a file\n')    
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
% USER DEFINED VARIABLE
wallVel        = 1.25
viscosity      = 1e-3
dpdx           = (101324 - 101325)/(1.0)

%__________________________________
% defaults
symbol   = {'+','*r','xg'}; 
pDir        = 1;
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
    uda = opt_value;
  elseif (strcmp(option,"-pDir") ) 
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


%________________________________
% do the Uintah utilities exist
[s0, r0]=unix('puda >& /dev/null');
[s1, r1]=unix('lineextract >& /dev/null');

if( s0 ~=0 || s1 ~= 0 )
  disp('Cannot execute uintah utilites puda, lineextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
end

%________________________________
%  extract the physical time
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = load('tmp');

if(ts == 999)  % default
  ts = length(physicalTime)
endif
%________________________________
%  extract the grid information from the uda file
c0 = sprintf('puda -gridstats %s >& tmp',uda); unix(c0);

[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp         |cut -d":" -f2 | tr -d "[]"');

resolution   = str2num(r1)
domainLength = str2num(r2)

% find the y direction
if(pDir == 1)
  yDir = 2;
elseif(pDir != 1)
  printf('\n\nWARNING: This script only works if the main axis of the problem is in the x direction\n\n');
  exit
end

% CONSTANTS
x_slice = resolution(pDir)-5;      % index where the data is extracted 

%__________________________________
% Extract the data from the uda
if(pDir == 1)
  startEnd = sprintf('-istart %i -1 0 -iend %i %i 0',x_slice,x_slice,resolution(yDir) );
elseif(pDir == 2)
%   to be filled in
elseif(pDir == 3)
%  to be filled in
end

c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o sim.dat -m %i  -uda %s','vel_CC >& /dev/null',L,ts-1,startEnd,mat,uda);
[s1, r1] = unix(c1);

% remove [] from velocity data
c2 = sprintf('sed ''s/\\[//g'' sim.dat | sed ''s/\\]//g'' >vel.dat');
[status2, result2]=unix(c2);

% import the velocity into array
vel     = load('vel.dat'); 
y_CC    = vel(:,yDir);
vel_sim = vel(:,3 + pDir);

%______________________________
% computes exact solution
nu = viscosity;
dy = y_CC(2) - y_CC(1)
h  = (domainLength(yDir) + dy)/2.0
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
y_ratio = y_CC/h;


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
unix('/bin/rm vel.dat sim.dat tmp');

%______________________________
% Plot the results
if (strcmp(makePlot,"true"))
  subplot(2,1,1),plot(vel_sim, y_CC, 'b:o;computed;', vel_exact, y_CC, 'r:+;exact;')
  xlabel('u velocity')
  ylabel('y')
  title('Combined Couette-Poiseuille Flow');
  grid on;
  
  subplot(2,1,2),plot(d,y_CC, 'b:+');
  hold on;
  xlabel('|u - u_{exact}|'); 
  ylabel('y');
  grid on;
  
  unix('/bin/rm Couette-Poiseuille.ps')
  print('Couette-Poiseuille.ps','-dps', '-FTimes-Roman:14')
  pause

end
