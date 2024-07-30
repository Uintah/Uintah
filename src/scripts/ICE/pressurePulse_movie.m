%______________________________________________________________________
%
% 07/30/24  --Todd
% This matLab/octave script generates 1D plots of the temperature,
% pressure, density and x-velocity.  It's used to compare different outlet boundary
% conditions for reflections.
%
%  For example:
%    pressurePulse_movie.m -uda1 pulse_Lodi_Li_scale_1.0.uda.000/ -uda2 pulse_2d_Neumann.uda.000/ -title " lodi vs neumann" -legend 'lodi;neumann'
%
%______________________________________________________________________

close all;
clear all;

graphics_toolkit gnuplot

function Usage
  printf('  pressurePulse_movie.m <options>\n')
  printf('    options:\n')
  printf('      -uda1  <udaFileName>      -  uda1  \n')
  printf('      -uda2  <udaFileName>      -  uda2  \n')
  printf('      -title  <string>          - title for the plots \n')
  printf('      -legend "string;string"   - legend for the plots \n')
end

%________________________________
%       bulletproofing
nargin = length(argv)
if (nargin != 8)
  printf( "   ERROR wrong number of input arguments: %i", nargin )
  Usage
  exit
endif

%________________________________
%       Parse user inputs

arg_list = argv ();

for i = 1:2:nargin
   option    = sprintf("%s", arg_list{i} );
   opt_value = sprintf("%s", arg_list{++i});

  if ( strcmp(option,"-uda1") )
    u1 = opt_value;
  elseif ( strcmp(option,"-uda2") )
    u2 = opt_value;
  elseif ( strcmp(option,"-title") )
    desc = opt_value
  elseif ( strcmp(option,"-legend") )
    legendText = strsplit(opt_value,";")
  end
end

udas={u1;u2}

%__________________________________
%     bulletproofing
s = isfolder(udas);

if( s != 1 )
  printf( "   ERROR one of the udas could not be found"  )
  udas
  Usage
  exit
endif


%__________________________________

pDir = 1;                 % Principal direction used for x axis
symbol = {'+','*r'};

xmin  = 0;                % x axis mininum
xmax  = 1;                % x axis maximum

                          % lineExtract start and stop and material
startEnd = '-istart 0 50 0 -iend 100 50 0 -m 0'

%________________________________
%       Do the Uintah utilities exist

s0 =system('which puda');
s1 =system('which lineextract');

if( s0 ~=0 || s1 ~= 0)
  disp('Cannot execute uintah utilites puda or lineextract');
  disp('  a) make sure you are in the right directory, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
  return;
endif

%__________________________________
%       clean up from previous run
system( '/bin/rm -rf movie');
system( 'mkdir movie');

%________________________________
%       extract the physical time for each output
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" > tmp',udas{1:1})
[status0, result0]=system(c0);

physicalTime  = importdata('tmp');
nDumps = length(physicalTime) - 1;

set(0, 'DefaultFigurePosition',[0,0,1024,768]);

%_________________________________
%       Loop over all the timesteps

for(ts = 1:nDumps )
  %ts = input('input timestep')
  time = sprintf('%d sec',physicalTime(ts+1));
                                      
  hFig = figure('visible','off');
                                      % gnuplot default text
  set(0, "defaulttextfontsize", 10)   % title
  set(0, "defaultaxesfontsize", 10)   % axes labels

  set(0, "defaulttextfontname", "Times-Roman")
  set(0, "defaultaxesfontname", "Times-Roman")
  set(0, "defaultlinemarkersize", 3)

  set(0, "defaultlinelinewidth", 0.5)

  %__________________________________
  %       Loop over udas
  for(i = 1:2)      %  loop over both udas
    uda = udas{i}

    %    pull out the data
    c1 = sprintf('lineextract -v rho_CC   -timestep %i %s -o rho.dat    -cellCoords  -uda %s',ts, startEnd, uda);
    c2 = sprintf('lineextract -v vel_CC   -timestep %i %s -o vel.dat    -cellCoords  -uda %s',ts, startEnd, uda);
    c3 = sprintf('lineextract -v temp_CC  -timestep %i %s -o temp.dat   -cellCoords  -uda %s',ts, startEnd, uda);
    c4 = sprintf('lineextract -v press_CC -timestep %i %s -o press.dat  -cellCoords  -m 0   -uda %s',ts,startEnd,uda)

    [status1, result1]=system(c1);
    [status2, result2]=system(c2);
    [status3, result3]=system(c3);
    [status4, result4]=system(c4);


    %     import the data into arrays
    press1{1,i}  = importdata('press.dat');
    temp1{1,i}   = importdata('temp.dat');
    rho1{1,i}    = importdata('rho.dat');
    vel1{1,i}    = importdata('vel.dat');

    system('/bin/rm *.dat');

    %__________________________________________________
    %   Plots
    %__________________________________________________
    %  temperature
    subplot(2,2,1), plot(temp1{1,i}(:,pDir),temp1{1,i}(:,4),symbol{i})

    axis([xmin xmax 200 1000])
    xlabel('x')
    ylabel('Temperature [K]')
    title(time);
    legend(legendText{1}, legendText{2})
    grid on;
    hold on;

    %______________________________
    %   pressure
    subplot(2,2,2),plot(press1{1,i}(:,pDir),press1{1,i}(:,4),symbol{i})

    axis([xmin xmax 10000 300000])
    xlabel('x')
    ylabel('Pressure[Pa]')
    title(desc);
    grid on;
    hold on;

    %_____________________________
    %  Density
    subplot(2,2,3), plot(rho1{1,i}(:,pDir),rho1{1,i}(:,4),symbol{i})

    axis([xmin xmax 0.5 3.0])
    xlabel('x')
    ylabel('Density [kg/m^3]')
    grid on;
    hold on;

    %____________________________
    %  velocity
    subplot(2,2,4), plot(vel1{1,i}(:,pDir), vel1{1,i}(:,4),symbol{i})

    axis([xmin xmax -125 125])
    ylabel('x-velocity [m/s]' )
    grid on;
    hold on;

 end

 hold off;
 name = sprintf('movie/movie.%04i.png', ts-1);
 saveas( hFig, name, "png" );
 close(hFig);
 
end   % loop over timesteps

%__________________________________
%    TotalMass plot
tMass = strcat (udas, ["/TotalMass.dat";"/TotalMass.dat"]);
s = isfile(tMass);

if (s != 1)
  printf( "   WARNING one of the TotalMass.dat files could not be found"  )
  printf( "   Now exiting...." )
  exit
endif


hFig = figure;
for(i = 1:2)      %  loop over both udas
  totalMass{1,i} = importdata( tMass{i} );
  
  plot(totalMass{1,i}(:,1), totalMass{1,i}(:,2),symbol{i});
  
  title(desc);
  legend(legendText{1}, legendText{2})
  xlabel('Time [s]' )
  ylabel('TotalMass [kg]' )
  grid on;
  hold on;
end

hold off;
name=sprintf('movie/TotalMass.png');
saveas( hFig, name, "png" );
close(hFig);
