#! /usr/bin/octave -qf
%_____________________________________________________________
% Function: compare_Riemann
clear all;
close all;
format short e;
function Usage
  printf('compare_Riemann.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n') 
  printf('  -test <1,2,3>       - name of the test youd like to run \n')                                                 
  printf('  -pDir <1,2,3>       - principal direction \n')                                                   
  printf('  -var  <press_CC, vel_CC, rho_CC, temp_CC> \n')                                                   
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
symbol   = {'b:+;computed;','*r','xg'}; 
pDir        = 999;
variable    = 'press_CC';
mat         = 0;
makePlot    = true;
ts          = 999;
output_file = 'L2norm';

arg_list = argv ();
for i = 1:2:nargin
   option    = sprintf("%s",arg_list{i} );
   opt_value = sprintf("%s",arg_list{++i});

  if ( strcmp(option,"-uda") )   
    uda = opt_value;
  elseif (strcmp(option,"-pDir") ) 
    pDir = str2num(opt_value);
  elseif (strcmp(option,"-var") )
    variable = opt_value;
  elseif (strcmp(option,"-mat") )
    mat = str2num(opt_value);
  elseif (strcmp(option,"-plot") )
    makePlot = opt_value;
  elseif (strcmp(option,"-test") )
    testNumber = str2num(opt_value);    
  elseif (strcmp(option,"-ts") )
    ts = str2num(opt_value);                  
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%________________________________
% do the Uintah utilities exist
%unix('setenv LD_LIBRARY /usr/lib')
[s0, r0]=unix('puda >&/dev/null');
[s1, r1]=unix('lineextract >&/dev/null');
[s2, r2]=unix('timeextract >&/dev/null');
[s3, r3]=unix('which exactRiemann >&/dev/null');
if( s0 ~=0 || s1 ~= 0 || s2 ~=0 || s3 ~=0)
  disp('Cannot execute Riemann or the Uintah utilites puda, timeextract lineextract or exactRiemann');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
  exit
end


%______________________________
% hardwired variables for 1 level problem
level = 0;
L = 1;
maxLevel = 1;

%________________________________
% extract the physical time 
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime  = load('tmp');

if(ts == 999)  % default
  ts = length(physicalTime)
endif

time = sprintf('%d sec',physicalTime(ts));
t = physicalTime(ts);

%________________________________
%  extract initial conditions and grid information from the uda file
c0 = sprintf('puda -gridstats %s >& tmp',uda); unix(c0);
[s,r0] = unix('grep -m1 dx: tmp| tr -d "dx:[]"');
[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp | tr -d "[:alpha:]:[],"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp         | tr -d "[:alpha:]:[],"');
dx           = str2num(r0);
resolution   = str2num(r1);         % this returns a vector
domainLength = str2num(r2);


%______________________________
% compute the exact solution for each variable
% The column format is
%  X    Rho_CC    vel_CC    Press_CC    Temp_CC
if( testNumber == 1 )
  inputFile = sprintf("test1.in");
elseif( testNumber == 2 )
  inputFile = sprintf("test2.in");
elseif( testNumber == 3 )
  inputFile = sprintf("test3.in");
elseif( testNumber == 4 )
  inputFile = sprintf("test4.in");
elseif( testNumber == 5 )
  inputFile = sprintf("test5.in");
endif

c = sprintf('exactRiemann %s %s %i %g', inputFile, 'exactSol', resolution(pDir), t)

[s, r] = unix(c);
exactSol = load('exactSol');
x_ex     = exactSol(:,1);

%______________________________
% Load the simulation solution into simSol

if(pDir == 1)
  startEnd = sprintf('-istart 0 0 0 -iend %i 0 0',resolution(pDir)-1);
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 0 0 -iend 0 %i 0',resolution(pDir)-1);
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 0 -iend 0 0 %i',resolution(pDir)-1);
end

variables = { 'rho_CC' 'vel_CC' 'press_CC' 'temp_CC'};

nrows = resolution(pDir);
ncols = length(variables);
susSol = zeros(nrows,ncols);
x      = zeros(nrows);

%__________________________________
% loop over all the variables and load them into susSol
for v=1:length(variables)
  c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o sim.dat -m %i  -uda %s >&/dev/null',variables(v),level,ts-1,startEnd,mat,uda);
  [s1, r1] = unix(c1);
  
  if ( strcmp(variables(v),'vel_CC'))         % for vel_CC
    % rip out [] from velocity data
    c2 = sprintf('cat sim.dat | tr -d "[]" > sim.dat2; mv sim.dat2 sim.dat');
    [r2, r2]=unix(c2);
    
    var = load('sim.dat');
    susSol(:,v) = var(:,3 + pDir);
    
  else                                        % all other variables
 
    var = load('sim.dat');
    susSol(:,v) = var(:,4); 
  endif
  
  x = var(:,pDir);
end

susSol;

%cleanup tmp files
unix('/bin/rm -f sim.dat');

% bulletproofing
test = sum (x - x_ex);
if(test > 1e-10)
  display('ERROR: compute_L2_norm: The results cannot be compared')
end

%__________________________________
% compute the difference/L-norm for each of the variables
d = zeros(nrows,ncols);

if(0)             %  skip sections of the domain.
  for v=1:length(variables)

    for c=1:length(x)                                                      
      d(c,v) = 0.0;                                                        

      if( x(c) < 0.7 || x(c) > 0.75)          %define the regions to skip  
        d(c,v) = ( susSol(c,v) .- exactSol(c,v+1) );                        
      end                                                                  
    end                                                                    
    
    L_norm(v) = dx(pDir) * sum( abs( d(:,v) ) );
  end
        
else              % include all of the cells in the calculation

  for v=1:length(variables)
    d(:,v) = ( susSol(:,v) .- exactSol(:,v+1) );
    L_norm(v) = dx(pDir) * sum( abs( d(:,v) ) );
  end
end


%__________________________________
% write L_norm to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  for v=1:length(variables)
    fprintf(fid,'%g ',L_norm(v));
  end
  fprintf(fid,'\n');
  fclose(fid);
end

%__________________________________
%write simulation data to a file
if (nargv > 0)
  fn = sprintf('sim_%g.dat',resolution(pDir));
  fid = fopen(fn, 'w');
  
  for c=1:length(x)
    fprintf(fid,'%g, ',x(c))
    for v=1:length(variables)
      fprintf(fid,'%g, ',susSol(c,v));
    end
    fprintf(fid, '\n')
  end
  
  fclose(fid);
end


% write the data to a file
%______________________________
if(makePlot)
  for v=1:length(variables)
    subplot(2,1,1), plot(x,susSol(:,v),symbol{L}, x_ex, exactSol(:,v+1),'r:;exact;');
    xlabel('x')

    tmp = sprintf('%s',variables(v));    
    ylabel(tmp)
    
    tmp = sprintf('Toro Test (%s) L1 norm: %f, time: %f', inputFile, L_norm(v),t);
    title(tmp);
    grid on;

    subplot(2,1,2),plot(x,d(:,v), 'b:+');
    ylabel('Difference'); 
    xlabel('x');
    grid on;
    fname = sprintf('%s_%i.eps',variables(v),resolution(pDir));
    print ( fname, '-deps');
   % pause
  end
  
  
  if(0)
    %______________________________
    % gradient of variable
    dx = abs(x(1) - x(2));
    gradVar   = gradient(susSol, dx);
    gradExact = gradient(exactSol,dx);
    figure(2)
    plot(x,gradVar,'b:;sus;',x,gradExact,'r:;exact;')

    xlabel('x')
    label = sprintf( 'Grad %s',variable);
    ylabel(label);
    grid on;
    pause
  endif
endif
