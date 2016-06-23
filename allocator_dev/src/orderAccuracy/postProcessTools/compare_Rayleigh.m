#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the rayleigh problem with and without dimensions
% Reference:  Incompressible Flow, by Panton pg 177
%
%  Example usage:
%  compare_Rayleigh.m -pDir 1 -mat 0 -plot false -o out.400.cmp -uda rayleigh_400.uda
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
% default user inputs
symbol   = {'+','*r','xg'}; 
pDir        = 999;
mat         = 0;
makePlot    = true;
ts          = 999;
output_file = 'L2norm';
L           = 0;
BIGNUM      = 1e7;

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

%  HARDWIRED CONSTANTS
viscosity      = 1e-4;
vel_CC_initial = 10.0;
rho_CC         = 1.1792946927374306;
%________________________________
% do the Uintah utilities exist
[s0, r0]=unix('puda > /dev/null 2>&1');
[s1, r1]=unix('lineextract > /dev/null 2>&1');

if( s0 ~=0 || s1 ~= 0 )
  disp('Cannot execute uintah utilites puda, lineextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
  quit(-1);
end

%________________________________
%  extract the physical time
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" > tmp 2>&1',uda);
[status0, result0]=unix(c0);
physicalTime  = load('tmp');

if(ts == 999)  % default
  ts = length(physicalTime)
endif
%________________________________
%  extract initial conditions and grid information from the uda file
c0 = sprintf('puda -gridstats %s > tmp 2>&1',uda); unix(c0);

[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp         |cut -d":" -f2 | tr -d "[]"');

resolution   = str2num(r1);
domainLength = str2num(r2);

% find the x and y directions

% for each axial direction different y directions are possible
% these scripts were built for each of the following planes
% when creating new new tests you may want to try to follow this convention

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
%compute the exact solution & L2Norm

%grabbing the data
xHalf = resolution(xDir)/2.0;

%sets the start and end cells for the line extract, depending on the plane
if(pDir == 1)
  startEnd = sprintf('-istart %i 0 0 -iend %i %i 0',xHalf,xHalf,resolution(yDir) -1);
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 %i 0 -iend 0 %i %i',xHalf,xHalf,resolution(yDir)-1);
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 %i -iend %i 0 %i',xHalf,resolution(yDir)-1,xHalf);
end

c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o vel.dat -m %i  -uda %s','vel_CC > /dev/null 2>&1',L,ts-1,startEnd,mat,uda);
[s1, r1] = unix(c1);

% import the data into arrays
vel  = load('vel.dat'); 
y_CC = vel(:,yDir);

% Add an offset to y_CC to compensate for the poor vel_CC boundary  condition in ICE
dy_2 = (y_CC(2) - y_CC(1) )/2;
y_CC_twk = y_CC + dy_2;
y_CC_twk = y_CC;

uvel = vel(:,3 + xDir);

% computes exact solution
time =physicalTime(ts);

vel_ratio_sim = uvel/vel_CC_initial;
nu = viscosity/rho_CC;

vel_ratio_exact =( 1.0 - erf( y_CC_twk/(2.0 * sqrt(nu * time)) ) );
vel_exact = vel_ratio_exact * vel_CC_initial;

clear d;
d = 0;
d = abs(vel_ratio_sim - vel_ratio_exact);
L2_norm = sqrt( sum(d.^2)/length(y_CC) );
length(y_CC);

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
% Plot the results from each timestep
% onto 2 plots
if (strcmp(makePlot,"true"))
  h = figure(1)
  subplot(2,1,1),plot(uvel, y_CC, 'b:o;computed;', vel_exact, y_CC_twk, 'r:+;exact;')
  xlabel('u velocity')
  ylabel('y')
  %axis('manual',[0 ,1e-4])
%  legend('computed');
  title('Rayleigh Problem');
  grid on;

%   subplot(3,1,2),plot(vel_ratio_exact,eta, 'b', vel_ratio_sim,eta, 'r:+');
%   hold on;
%   xlabel('u/U0'); 
%   ylabel('eta');
%   %legend('exact', 'computed');
%   grid on;
  subplot(2,1,2),plot(d,y_CC, 'b:+');
  hold on;
  xlabel('u - u_exact'); 
  ylabel('y');
  grid on;
  c1 = sprintf('%i_%i.jpg',resolution(yDir),yDir);
  %saves the plot to an output file
  %naming convention for the output file (resolution,yDir.jpg)
  print (c1,'-djpg')
end
