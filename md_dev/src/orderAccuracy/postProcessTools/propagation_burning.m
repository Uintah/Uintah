#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the propagation of the convective burning
% vs the time
%
%  Example usage:
%  propagation_burning.m -pDir 1 -mat 0 -o out.400.cmp 
%  run propagation_burning.m in the x direction using material zero (material isnt used)
%_________________________________
clear all;
close all;
format short e;

function Usage
  printf('propagation_burning.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                  
  printf('  -pDir <1,2,3>       - principal direction \n')                                                   
  printf('  -mat                - material index \n') 
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
pDir        = 1;
mat         = 0;
makePlot    = true;
rho_CC      = 1832;

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
n_ts = length(physicalTime);

%________________________________
%  extract grid information from the uda file
c0 = sprintf('puda -gridstats %s >& tmp',uda); unix(c0);

[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp         |cut -d":" -f2 | tr -d "[]"');

resolution   = str2num(r1);
domainLength = str2num(r2);
deltaX = domainLength./resolution;
dx = -9;
SA = -9;

% find the y direction
if(pDir == 1)
  dx   = deltaX(1);
  dy   = deltaX(2);
  SA   = dx * dy;
  yDir = 2;
elseif(pDir == 2)     %FIX ME!!!
  dx = deltaX(2)
  yDir = -9;
elseif(pDir == 3)
  dx = deltaX(3)
  yDir = -9;
end

%__________________________________
% Deteriming the starting and ending cells
% to extract data from.
if(pDir == 1)
  startEnd = sprintf('-istart 0 0 0 -iend %i 1 0',resolution(pDir)-1);
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 0 0 -iend 0 %i 0',resolution(pDir)-1);
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 0 -iend 0 0 %i',resolution(pDir)-1);
end

%__________________________________
% Loop over all ofthe timesteps and extract the "burning" data
% write it to a file.

fid = fopen('MaxX.dat', 'a');

for ts=1:n_ts
  c1       = sprintf('lineextract -cellCoords -v burning -timestep %i %s -uda %s -o sim.dat >&/dev/null',ts-1,startEnd,uda);
  [s1, r1] = unix(c1);
  
  tmp     = load('sim.dat');
  x_CC    = tmp(:,pDir);
  burn    = tmp(:,4);
  nLen_xCC = length(x_CC);
  ix       = 0;
  
  for i=1:nLen_xCC
    if( burn(i) == 1 && i > ix ) 
      ix = i;
    endif
  end
  ix
  
  if( ix == 0 )    % If there is no burning
    fprintf( fid,"%e %15.16f\n", physicalTime(ts), -9 );
  else             % if there is burning
    fprintf( fid,"%e %15.16f\n", physicalTime(ts), x_CC(ix) );
  end    
  
end

fclose(fid);

%______________________________
% Load MaxX and Time
X      = load('MaxX.dat');
maxX   = X(:,2);
time_X = X(:,1);
nLen_X = length(maxX);

maxX
time_X

% Plot it up 
%________________________________
 plot(time_X, maxX, 'b:o;;')
  xlabel('Time (s)')
  ylabel('maxX')
  title('maxX vs Time')
  grid on;
 unix('/bin/rm Time_MaxX.ps >&/dev/null');
  print('Time_MaxX.ps','-dps', '-FTimes-Roman:14');
 
  c2 = sprintf('mv Time_MaxX.ps %s.000',uda);
  unix(c2);
  
  c3 = sprintf('mv MaxX.dat %s.000',uda);
  unix(c3);

% Determine Velocity  
%_______________________________
velocity = zeros(nLen_X,1);

velocity(1) =0;   % first data point
tOld    = 0;   %initalizing tOld
for c=1:nLen_X-1
  if( maxX(c+1) > maxX(c) )
    dt             = time_X(c+1) - tOld;
    dX             = maxX(c+1) - maxX(c);
    velocity(c+1)  = dX/dt;
    tOld           = time_X(c+1);
  endif
  if( maxX(c+1) == maxX(c))
    velocity(c+1)  = velocity(c);
  endif  
end 


% Plot it up 
%________________________________
figure(2);
  plot(time_X, velocity, 'bo;;')
    xlabel('Time (s)')
    ylabel('Instantaneous Velocity (m/s)')
    title('Instantaneous Velocity vs Time');
    grid on;

unix('/bin/rm InstantaneousTime_Velocity.ps >&/dev/null');
  print('InstantaneousTime_Velocity.ps','-dps', '-FTimes-Roman:14');
  
  c4 = sprintf('mv InstantaneousTime_Velocity.ps %s.000',uda);
  unix(c4);

  c5          = sprintf('cp %s.000/InstantaneousTime_Velocity.ps %s_InstantaneousTime_Velocity.ps',uda,uda);
  [s5, r5]    = unix(c5);
  
fid = fopen('vel.dat', 'a');
for i=1:nLen_X
  fprintf(fid," %16.15f %16.15f\n", time_X(i), velocity(i));
end

fclose(fid);
c6 = sprintf('mv vel.dat %s.000',uda);
  unix(c6);
