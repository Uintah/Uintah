#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the rayleigh problem with and without dimensions
% Reference: Ghia et al. - 1982 - High-Re solutions for incompressible flow using the Naiver-Stokes equations and multigrid method 
%
%  Example usage:
%  compare_Rayleigh.m -pDir 1 -mat 0 -plot false -o out.400.cmp -uda rayleigh_400.uda
%_________________________________

clear all;
close all;
format short e;

function Usage
  printf('cavityFlow.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                  
  printf('  -aDir <1,2,3>       - axial direction \n')                                                   
  printf('  -mat                - material index \n')                                                        
  printf('  -plot <true, false> - produce a plot \n')                                                        
  printf('  -ts                 - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>          - Dump the output (L2Error) to a file\n') 
  printf('  -Re <Reynolds Number> - Re may be 100,400,1000,3200,5000,7500,10000')                                    
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
myPath   = mfilename ("fullpathext");
srcPath  = readlink( myPath );
funcPath = strcat( fileparts (srcPath), "/functions" );
addpath( funcPath )

%__________________________________
% default user inputs
symbol   = {'+','*r','xg'}; 
pDir        = 1;
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
    output_file = opt_value    
  elseif (strcmp(option,"-Re") )
    Re = str2num(opt_value);
  end                                      
end

%__________________________________
% extract time and grid info on this level
tg_info = getTimeGridInfo( uda, ts, L );

ts           = tg_info.ts;
resolution   = tg_info.resolution;

% find the y direction

% for each axial direction different y directions are possible
% these scripts were built for each of the following planes
% when creating new new tests you may want to try to follow this convention

% x-y 
if(pDir == 1)
  yDir = 2;
  xDir = 1;
% x-z
elseif(pDir == 2)
  yDir = 3;    
  xDir = 2;  
% y-z      
elseif(pDir == 3)
  yDir = 1;
  xDir = 3;           
end

%__________________________________
%compute the exact solution & L2Norm

%grabbing the data
%grabs the v velocity along the y direction in the middle of the x direction
xHalf = resolution(xDir)/2.0;
yHalf = resolution(yDir)/2.0;
%setting the starting and end cells for the line extract
if(pDir == 1)
  startEnd = sprintf('-istart %i -1 0 -iend %i %i 0',xHalf,xHalf,resolution(yDir));   
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 %i -1 -iend 0 %i %i',xHalf,xHalf,resolution(yDir));    
elseif(pDir == 3)
  startEnd = sprintf('-istart -1 0 %i -iend %i 0 %i',xHalf,resolution(yDir),xHalf);     
end

c1 = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o velV.dat -m %i -uda %s > /dev/null 2>&1',L,ts-1,startEnd,mat,uda)
[s1, r1] = unix(c1);

% import the data into arrays
vel  = load('velV.dat'); 
y_CC = vel(:,yDir);
dy_2 = (y_CC(2) - y_CC(1) )/2;
y_CC_twk = y_CC + dy_2;
y_CC_twk = y_CC;
uvel = vel(:,3 + xDir);
y_CC = y_CC/y_CC(length(y_CC));
velInitial = uvel(length(uvel));
uvel = uvel/velInitial;

%Redoing everything in the x direction and v velocity
if(pDir == 1)
  startEnd = sprintf('-istart -1 %i 0 -iend %i %i 0',yHalf,resolution(xDir),yHalf);   
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 -1 %i -iend 0 %i %i',yHalf,resolution(xDir),yHalf);   
elseif(pDir == 3)
  startEnd = sprintf('-istart %i 0 -1 -iend %i 0 %i',yHalf,yHalf,resolution(xDir));    
end

c1 = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o velH.dat -m %i -uda %s > /dev/null 2>&1',L,ts-1,startEnd,mat,uda);
[s1, r1] = unix(c1);

% import the data into arrays
vel2  = load('velH.dat');
x_CC = vel2(:,xDir);
dx_2 = (x_CC(2) - x_CC(1) )/2;
x_CC_twk = x_CC + dx_2;
x_CC_twk = x_CC;
vvel = vel2(:,3 + yDir);
x_CC = x_CC/x_CC(length(x_CC));
vvel = vvel/velInitial;


%data compared against 

%This table is the data from Ghia et al. - 1982
%This is the u velocity data and is formatted in this way
% [cell number,(y/h),Re(100),Re(400),Re(1000),Re(3200),Re(5000),Re(7500),Re(10000)]
% Where cell number is the cell number the data was extracted from
% y is the height from the bottom of the square, h is the total heigh of the square
% Re(x) is the nondimensional velocity at Re and y.
% the nondimensional velocity is u/uLid

col = 4;
if(Re == 100)
  col = 3;
elseif(Re == 400)
  col = 4;
elseif(Re == 1000)
  col = 5;
elseif(Re == 3200)
  col = 6;
elseif(Re == 5000)
  col = 7;
elseif(Re == 7500)
  col = 8;
elseif(Re == 10000)
  col = 9;
end

udata = [ \
129, 1.0000 ,  1.00000 ,  1.00000 , 1.00000 , 1.00000 , 1.00000  , 1.00000 ,  1.00000;\
126, 0.9766 ,  0.84123 ,  0.75837 , 0.65928 , 0.53236 , 0.48223  , 0.47244 ,  0.47221;\
125, 0.9688 ,  0.78871 ,  0.68439 , 0.57492 , 0.48296 , 0.46120  , 0.47048 ,  0.47783;\
124, 0.9609 ,  0.73722 ,  0.61756 , 0.51117 , 0.46547 , 0.45992  , 0.47323 ,  0.48070;\
123, 0.9531 ,  0.68717 ,  0.55892 , 0.46604 , 0.46101 , 0.46036  , 0.47167 ,  0.47804;\
110, 0.8516 ,  0.23151 ,  0.29093 , 0.33304 , 0.34682 , 0.33556  , 0.34228 ,  0.34635;\
 95, 0.7344 ,  0.00332 ,  0.16256 , 0.18719 , 0.19791 , 0.20087  , 0.20591 ,  0.20673;\
 80, 0.6172 , -0.13641 ,  0.02135 , 0.05702 , 0.07156 , 0.08183  , 0.08342 ,  0.08344;\
 65, 0.5000 , -0.20581 , -0.11477 ,-0.06080 ,-0.04272 ,-0.03039  ,-0.03800 ,  0.03111;\
 59, 0.4531 , -0.21090 , -0.17119 ,-0.10648 ,-0.86636 ,-0.07404  ,-0.07503 , -0.07540;\
 37, 0.2813 , -0.15662 , -0.32726 ,-0.27805 ,-0.24427 ,-0.22855  ,-0.23176 , -0.23186;\
 23, 0.1719 , -0.10150 , -0.24299 ,-0.38289 ,-0.34323 ,-0.33050  ,-0.32393 , -0.32709;\
 14, 0.1016 , -0.06434 , -0.14612 ,-0.29730 ,-0.41933 ,-0.40435  ,-0.38324 , -0.38000;\
 10, 0.0703 , -0.04775 , -0.10338 ,-0.22220 ,-0.37827 ,-0.43643  ,-0.43025 , -0.41657;\
  9, 0.0625 , -0.04192 , -0.09266 ,-0.20196 ,-0.35344 ,-0.42901  ,-0.43590 , -0.42537;\
  8, 0.0547 , -0.03717 , -0.08186 ,-0.18109 ,-0.32407 ,-0.41165  ,-0.43154 , -0.42735;\
  1, 0.0000 ,  0.00000 ,  0.00000 , 0.00000 , 0.0000  , 0.00000  , 0.00000 ,  0.00000 ] ;

y_CCE = udata(:,2);
uvelE = udata(:,col);

%This is the v velocity data and is formatted in this way
% [cell number,(x/L),Re(100),Re(400),Re(1000),Re(3200),Re(5000),Re(7500),Re(10000)]
% Where cell number is the cell number the data was extracted from
% x is the distance from the x- side of the square, L is the total length of the square
% Re(Re) is the nondimensional velocity at Re and x.
% the nondimensional velocity is u/uLid


vdata = [ \
129, 1.0000 , 0.00000 , 0.00000 ,  0.00000 ,  0.00000 ,  0.00000  , 0.00000 ,  0.00000;\
125, 0.9688 ,-0.05906 ,-0.12146 , -0.21388 , -0.39017 , -0.49774 , -0.53858 , -0.54302;\
124, 0.9609 ,-0.07391 ,-0.15663 , -0.27669 , -0.47425 , -0.55069 , -0.55216 , -0.52987;\
123, 0.9531 ,-0.08864 ,-0.19254 , -0.33714 , -0.52357 , -0.55408 , -0.52347 , -0.49099;\
122, 0.9453 ,-0.10313 ,-0.22847 , -0.39188 , -0.54053 , -0.52876 , -0.48590 , -0.45863;\
117, 0.9063 ,-0.16914 ,-0.23827 , -0.51550 , -0.44307 , -0.41442 , -0.41050 , -0.41496;\
111, 0.8594 ,-0.22445 ,-0.44993 , -0.42665 , -0.37401 , -0.36214 , -0.36213 , -0.36737;\
104, 0.8047 ,-0.24533 ,-0.38598 , -0.31966 , -0.31184 , -0.30018 , -0.30448 , -0.30719;\
 65, 0.5000 , 0.05454 , 0.05188 ,  0.02526 ,  0.00999 ,  0.00945 ,  0.00824 ,  0.00831;\
 31, 0.2344 , 0.17527 , 0.30174 ,  0.32235 ,  0.28188 ,  0.27280 ,  0.27348 ,  0.27224;\
 30, 0.2266 , 0.17507 , 0.30203 ,  0.33075 ,  0.29030 ,  0.28066 ,  0.28117 ,  0.28003;\
 21, 0.1563 , 0.16077 , 0.28124 ,  0.37095 ,  0.37119 ,  0.35368 ,  0.35060 ,  0.35070;\
 13, 0.0938 , 0.12317 , 0.22965 ,  0.32627 ,  0.42768 ,  0.42951 ,  0.41824 ,  0.41487;\
 11, 0.0781 , 0.10890 , 0.20920 ,  0.30353 ,  0.41906 ,  0.43648 ,  0.43564 ,  0.43124;\
 10, 0.0703 , 0.10091 , 0.19713 ,  0.29012 ,  0.40917 ,  0.43329 ,  0.44030 ,  0.43733;\
  9, 0.0625 , 0.09233 , 0.18360 ,  0.27485 ,  0.39560 ,  0.42447 ,  0.43979 ,  0.43983;\
  1, 0.0000 , 0.00000 , 0.00000 ,  0.00000 ,  0.00000 ,  0.00000 ,  0.00000 ,  0.00000 ] ;

x_CCE = vdata(:,2);
vvelE = vdata(:,col);

%Compute the L2 norm
for i = 1:length(uvelE)
  count = 1;
  while (y_CCE(i) > y_CC(count))
    count = count + 1 ;
  end
  uvelC(i) = uvel(count);
end

for i = 1:length(vvelE)
  count = 1;
  while (x_CCE(i) > x_CC(count))
    count = count + 1;
  end
  vvelC(i) = vvel(count);
end

d = 0;
e = 0;
d = abs(uvelC' - uvelE);
e = abs(vvelC' - vvelE);
L2d = sqrt(sum(d.^2)/length(y_CCE));
L2e = sqrt(sum(e.^2)/length(x_CCE));
L2_norm = L2d + L2e
length(y_CC);

nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',L2_norm);
  fclose(fid);
end  
  
  
% cleanup 
unix('/bin/rm velV.dat velH.dat sim.dat tmp');


%______________________________
% Plot the results from each timestep
% onto 2 plots
if (strcmp(makePlot,"true"))
  subplot(2,1,1),plot(uvel, y_CC, 'b:o;computed;',uvelE,y_CCE, 'r:+;exact;');
  xlabel('u/umax')
  ylabel('y/h')
  title('Cavity Flow');
  grid on;
  
  subplot(2,1,2),plot(x_CC,vvel, 'b:o;computed;',x_CCE,vvelE, 'r:+;exact;');
  hold on;
  xlabel('x/L'); 
  ylabel('v/vmax');
  grid on;
  c1 = sprintf('%i_%i_%i.jpg',resolution(pDir),pDir,Re);
  %Prints the plot to a file 
  print(c1,'-djpg')
  
  %This section outputs data files which can be read into a gnuplot script 
  %that overlays each plot for each plane
  %To use un comment this section and once the tests completes
  %move all the .dat files to the same directory as the gnuplot script "scriptname"
  %and run the script. The script may require tweaking depending on some factors of the input
  %file that you may have changed due to the naming convention of (resolution_axialDir_Re.dat)
  
  %{
  c2 = sprintf('u%i_%i_%i.dat',resolution(pDir),pDir,Re);
  uout = zeros(length(uvel),2);
  uout(:,1) = y_CC;
  uout(:,2) = uvel;
  save("-text",c2,"uout")
  
  c3 = sprintf('uE_%i.dat',Re);
  uoutE = zeros(length(uvelE),2);
  uoutE(:,1) = y_CCE;
  uoutE(:,2) = uvelE;
  save("-text",c3,"uoutE")
  
  c4 = sprintf('v%i_%i_%i.dat',resolution(pDir),pDir,Re);
  vout = zeros(length(vvel),2);
  vout(:,1) = x_CC;
  vout(:,2) = vvel;
  save("-text",c4,"vout")
  
  c5 = sprintf('vE_%i.dat',Re);
  voutE = zeros(length(vvelE),2);
  voutE(:,1) = x_CCE;
  voutE(:,2) = vvelE;
  save("-text",c5,"voutE")
  %}
end





