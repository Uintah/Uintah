#! /usr/bin/octave -qf
%_____________________________________________________________
% Function: compare_Riemann
%   This function depends on the executable "exactRiemann" which is part of
%   Numerica.  You need to compile it before you can run this script.
%
%    cd src/parametricStudies/postProcessTools/Numerica
%    ./compile
%
%______________________________________________________________________


clear all;
close all;
format short e;
function Usage
  printf('compare_Riemann.m <options>\n')
  printf('options:\n')
  printf('  -uda  <udaFileName> - name of the uda file \n')
  printf('  -test <1,2,3>       - name of the test to run \n')
  printf('  -pDir <1,2,3>       - principal direction \n')
  printf('  -mat                - material index \n')
  printf('  -plot <true, false> - produce a plot \n')
  printf('  -ts                 - Timestep to compute L2 error, default is the last timestep\n')
  printf('  -o <fname>          - Dump the output (L2Error) to a file\n')
end

%________________________________
%   Parse User inputs
%echo
nargin = length(argv);
if (nargin == 0)
  Usage
  exit
endif

%__________________________________
%   update paths
myPath   = readlink( mfilename ("fullpathext") );
scriptPath = fileparts(myPath);
uintah   = textread( 'scriptPath', '%s','endofline', '\n' );
path     = sprintf( '%s:%s/functions:%s/Numerica', uintah{:}, scriptPath, scriptPath );
addpath( path )

%   path used by octave:unix command
unixPath = sprintf( '%s:%s', EXEC_PATH(), path );
EXEC_PATH ( unixPath )

%__________________________________
%   default user inputs
symbol   = {'b:+;computed;','*r','xg'};
pDir        = 999;
mat         = 0;
makePlot    = true;
ts          = 999;
output_file = 'Errors.dat';

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

%______________________________
% hardwired variables for 1 level problem
level = 0;
L = 1;

%________________________________
% does code for computing exact solution exist
[s, r]=unix('which exactRiemann');

if( s == 1 )
  printf("__________________________________ERROR\n")
  printf(" Cannot find the executable exactRiemann\n \
           To compile the code:\n \
           cd %s/Numerica\n \
           ./compile\n \
  Now exiting.\n", scriptPath);
  printf("__________________________________\n")
  quit(-1);
end

%__________________________________
% extract time and grid info on this level
tg_info = getTimeGridInfo( uda, ts, level );

ts           = tg_info.ts;
resolution   = tg_info.resolution;
t            = tg_info.physicalTime;
dx           = tg_info.dx;

time = sprintf('%d sec', t);

%______________________________
% compute the exact solution for each variable
% The column format is
%  X    Rho_CC    vel_CC    Press_CC    Temp_CC
if( testNumber == 1 )
  inputFile = sprintf( "test1.in" );
elseif( testNumber == 2 )
  inputFile = sprintf( "test2.in" );
elseif( testNumber == 3 )
  inputFile = sprintf( "test3.in" );
elseif( testNumber == 4 )
  inputFile = sprintf( "test4.in" );
elseif( testNumber == 5 )
  inputFile = sprintf( "test5.in" );
endif

cmd = sprintf('exactRiemann %s %s %i %g', inputFile, 'exactSol', resolution(pDir), t);

printf( "       Now computing the exact solution (%s)\n", cmd)
[s, r]  = unix(cmd);

if( s != 0 )
  printf("__________________________________ERROR\n")
  printf(" There was a problem computing the exact solution executable exactRiemann\n \
           %s\n \
  Now exiting.\n", cmd);
  printf("__________________________________\n")
  quit(-1);
end


exactSol = load('exactSol');
x_ex     = exactSol(:,1);

%______________________________
%   Load the simulation solution into simSol

if(pDir == 1)
  startEnd = sprintf('-istart 0 0 0 -iend %i 0 0',resolution(pDir)-1);
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 0 0 -iend 0 %i 0',resolution(pDir)-1);
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 0 -iend 0 0 %i',resolution(pDir)-1);
end

variables = {"rho_CC"; "vel_CC"; "press_CC"; "temp_CC"; "mach"};  % don't change the order of these unless you also alter
                                                          % Numerica/e1rpexModified.f

nrows = resolution(pDir);
ncols = length(variables);
susSolution = zeros(nrows,ncols);
x      = zeros(nrows);

%__________________________________
%   loop over all the variables and load them into susSolution
printf( "       Now extracting the variables from the uda %s", uda)

for v=1:length(variables)

  var = variables{v};

  c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o sim.dat -m %i  -uda %s > /dev/null 2>&1',var,level,ts-1,startEnd,mat,uda);
  [s1, r1] = unix(c1);

  if ( strcmp(var,'vel_CC') )         % for vel_CC

    % rip out [] from velocity data
    c2 = sprintf('cat sim.dat | tr -d "[]" > sim.dat2; mv sim.dat2 sim.dat');
    [r2, r2]=unix(c2);

    vel = load('sim.dat');
    susSolution(:,v) = vel(:,3 + pDir);

  else                                        % all other variables

    var = load('sim.dat');
    susSolution(:,v) = var(:,4);
  endif

  x = var(:,pDir);
end

susSolution;

%   cleanup tmp files
unix('/bin/rm -f sim.dat');

%   bulletproofing
test = sum( x - x_ex );
if( abs(test) > 1e-10)
  display('ERROR: compute_L2_norm: The results cannot be compared since the x locations are not the same')
end

%__________________________________
%   compute the L2 norm and L Infinity norms for each of the variables
difference = zeros(nrows,ncols);

if(0)             %  skip sections of the domain.
  for v=1:length(variables)

    for c=1:length(x)
      difference(c,v) = 0.0;

      if( x(c) < 0.7 || x(c) > 0.75)          %define the regions to skip
        difference(c,v) = ( susSolution(c,v) .- exactSol(c,v+1) );
      end
    end
    L2norm(v)    = sqrt( sum(difference.^2)/length(difference) )
    LInfinity(v) = max(difference)
  end

else              % include all of the cells in the calculation

  for v=1:length(variables)
    difference   = ( susSolution(:,v) .- exactSol(:,v+1) );
    L2norm(v)    = sqrt( sum(difference.^2)/length(difference)  );
    LInfinity(v) = max(difference);
  end
end


%__________________________________
%   write L_norm to a file
nargv = length(output_file);
if (nargv > 0)

  output_Lnorm( 'L2norm.dat',        variables, resolution(pDir), L2norm )
  output_Lnorm( 'LInfinityNorm.dat', variables, resolution(pDir), LInfinity )

end


%__________________________________
%   write simulation data to a file
susResults = sprintf('sim_%g.dat',resolution(pDir));
if (nargv > 0)
  fid = fopen(susResults, 'w');

  for c=1:length(x)
    fprintf(fid,'%E, ',x(c))
    for v=1:length(variables)
      fprintf(fid,'%E, ',susSolution(c,v));
    end
    fprintf(fid, '\n')
  end

  fclose(fid);
end

%______________________________
%   Plot the results.  Use gnuplot since octave plotting routines are difficult
if(makePlot)

  Title = sprintf( "\"Riemann Problem: %i, resolution: %i, pDir: %i \"",testNumber, resolution(pDir), pDir );
  cmd = sprintf( " gnuplot -c %s %s %s %i %s", "plotResults.gp", susResults, "exactSol", resolution(pDir), Title );
  printf( "       Now plotting the exact solution vs ICE's solution (%s)\n", cmd)
  unix (cmd);

endif
