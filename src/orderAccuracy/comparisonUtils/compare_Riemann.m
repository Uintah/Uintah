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
nargin = length(argv)
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
   option    = sprintf("%s",arg_list{i} )
   opt_value = sprintf("%s",arg_list{++i})

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
    testNumber = opt_value;    
  elseif (strcmp(option,"-ts") )
    ts = str2num(opt_value);                  
  elseif (strcmp(option,"-o") )  
    output_file = opt_value;    
  end                                      
end

%________________________________
% do the Uintah utilities exist
%unix('setenv LD_LIBRARY /usr/lib')
[s0, r0]=unix('puda');
[s1, r1]=unix('lineextract');
[s2, r2]=unix('timeextract');
[s3, r3]=unix('which exactRiemann');
if( s0 ~=0 || s1 ~= 0 || s2 ~=0 || s3 ~=0)
  disp('Cannot execute Riemann or the Uintah utilites puda, timeextract lineextract or exactRiemann');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
  exit
end


%______________________________
% mapping between variable name and column in exact solution data
if( strcmp(variable, 'rho_CC'))
  col = 2;
elseif( strcmp(variable,'vel_CC') )
  col = 3;
elseif( strcmp(variable,'press_CC') )
  col = 4;
elseif( strcmp(variable,'temp_CC') )
  col = 5;
else
  display('Error: unknown variable;')
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
unix('grep -m1 dx: tmp| tr -d "dx:[]"');

[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp | tr -d "[:alpha:]:[],"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp         | tr -d "[:alpha:]:[],"');

resolution   = str2num(r1);
domainLength = str2num(r2);

%______________________________
% compute the exact solution
%if( strcmp(testNumber,'1')
  %inputFile = sprintf("test1.in")
%endif

c = sprintf('exactRiemann %s %s %i', 'test1.in', 'exactSol', resolution(pDir))

[s, r] = unix(c);
exactSol_tmp = load('exactSol');
exactSol = exactSol_tmp(:,col);
x_ex     = exactSol_tmp(:,1);

%______________________________
% compute L2norm
if(pDir == 1)
  startEnd = sprintf('-istart 0 0 0 -iend %i 0 0',resolution(pDir)-1)
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 0 0 -iend 0 %i 0',resolution(pDir)-1)
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 0 -iend 0 0 %i',resolution(pDir)-1)
end

c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o sim.dat -m %i  -uda %s',...
  variable,level,ts-1,startEnd,mat,uda)
[s1, r1] = unix(c1)


var{1,L} = load('sim.dat');
x      = var{1,L}(:,pDir);
susSol = var{1,L}(:,4);

%cleanup
%unix('/bin/rm -f sim.dat tmp?.dat vel?.dat');

test = sum (x - x_ex);
if(test > 1e-10)
  display('ERROR: compute_L2_norm: The results cannot be compared')
end

clear d;          % d is the difference
d = 0; 

for( i = 1:length(x))
  d(i) = (susSol(i) - exactSol(i));
end

L2_norm = sqrt( sum(d.^2)/length(d) )

% write L2_norm to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',L2_norm);
  fclose(fid);
end
%______________________________
if(makePlot)
  clear temp1;
  subplot(2,1,1), plot(x,susSol,symbol{L}, x_ex, exactSol,'r:;exact;')
  xlabel('x')
  ylabel(variable)
  title('shockTube Problem');
  grid on;
  
  subplot(2,1,2),plot(x,d, 'b:+');
  hold on;
  ylabel('Difference'); 
  xlabel('x');
  grid on;
  pause
  if(1)
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
  end
end
