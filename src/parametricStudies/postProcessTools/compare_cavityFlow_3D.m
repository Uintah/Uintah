#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the velocity profile and computes
% the L2norm of the rayleigh problem with and without dimensions
% Reference: Ghia et al. - 1982 - High-Re solutions for incompressible flow using the Naiver-Stokes equations and multigrid method 
%
%  Example usage:
%  compare_cavityFlow_3D.m -aDir 1 -mat 0 -plot false -Re 3200 -o out.3200.cmp -uda cavityFlow_3D.uda
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
domainLength = tg_info.domainLength;
time         = tg_info.physicalTime;


% find the y direction

% for each axial direction different y directions are possible
% these scripts were built for each of the following planes
% when creating new new tests you may want to try to follow this convention

% x-y 
if(pDir == 1)
  zDir = 3;
  yDir = 2;
  xDir = 1;
% x-z
elseif(pDir == 2)
  yDir = 3;    
  xDir = 2;
  zDir = 1;  
% y-z      
elseif(pDir == 3)
  yDir = 1;
  zDir = 2;
  xDir = 3;           
end

%__________________________________
%compute the exact solution & L2Norm

%grabbing the data
%grabs the v velocity along the y direction in the middle of the x direction
xHalf = resolution(xDir)/2.0;
yHalf = resolution(yDir)/2.0;
zHalf = resolution(zDir)/2.0;
%setting the starting and end cells for the line extract
if(pDir == 1)
  startEnd = sprintf('-istart %i -1 %i -iend %i %i %i',xHalf,zHalf,xHalf,resolution(yDir),zHalf);   
elseif(pDir == 2)
  startEnd = sprintf('-istart %i %i -1 -iend %i %i %i',zHalf,xHalf,zHalf,xHalf,resolution(yDir));    
elseif(pDir == 3)
  startEnd = sprintf('-istart -1 %i %i -iend %i %i %i',zHalf,xHalf,resolution(yDir),zHalf,xHalf);     
end

c1 = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o velV.dat -m %i -uda %s > /dev/null 2>&1',L,ts-1,startEnd,mat,uda);
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
  startEnd = sprintf('-istart -1 %i %i -iend %i %i %i',yHalf,zHalf,resolution(xDir),yHalf,zHalf);   
elseif(pDir == 2)
  startEnd = sprintf('-istart %i -1 %i -iend %i %i %i',zHalf,yHalf,zHalf,resolution(xDir),yHalf);   
elseif(pDir == 3)
  startEnd = sprintf('-istart %i %i -1 -iend %i %i %i',yHalf,zHalf,yHalf,zHalf,resolution(xDir));    
end

c1 = sprintf('lineextract -v vel_CC -l %i -cellCoords -timestep %i %s -o velH.dat -m %i -uda %s > /dev/null 2>&1',L,ts-1,startEnd,mat,uda)
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

% A dynamic mixed subgrid-scale model and its application to turbulent recirculating flows
% Yan Zang, Robert L. Street, and Jeffrey R. Koseff
% pg 3190-3192, aquired with Data Thief

%Data is ordered as follows (x,u) 
%Where x is the nondimensional length with a cordinate system of 0,0 being in the center of the square
%u is the nondimension velocity u/uLid

if (Re == 3200)
  udata = [ \
  1.0000 ,  1.0000 ;\
  0.9866 ,  0.7210 ;\
  0.9676 ,  0.5408 ;\
  0.9240 ,  0.2479 ;\      
  0.8940 ,  0.1813 ;\      
  0.8653 ,  0.1589 ;\      
  0.7966 ,  0.1309 ;\      
  0.7351 ,  0.1104 ;\      
  0.6660 ,  0.0986 ;\      
  0.6026 ,  0.0778 ;\      
  0.4037 ,  0.0528 ;\      
  0.2076 ,  0.0338 ;\      
  0.0030 ,  0.0030 ;\      
 -0.1989 , -0.0101 ;\      
 -0.4007 , -0.0300 ;\      
 -0.5968 , -0.0901 ;\      
 -0.6604 , -0.1521 ;\      
 -0.7296 , -0.2016 ;\      
 -0.8023 , -0.2489 ;\      
 -0.8681 , -0.2656 ;\      
 -0.9142 , -0.2487 ;\      
 -0.9315 , -0.2197 ;\      
 -0.9569 , -0.1770 ;\      
 -0.9713 , -0.1264 ;\      
 -0.9857 , -0.0506 ;\      
 -1.0000 ,  0.0000] ;
 
elseif (Re == 7500)
  udata = [ \
  1.0000 ,  1.0000 ;\
  0.9862 ,  0.6010 ;\
  0.9806 ,  0.5299 ;\
  0.9694 ,  0.3746 ;\      
  0.9607 ,  0.2585 ;\      
  0.9489 ,  0.1605 ;\      
  0.9325 ,  0.1031 ;\      
  0.9017 ,  0.0763 ;\      
  0.5844 ,  0.0363 ;\      
  0.2748 ,  0.0150 ;\      
  0.0000 ,  0.0000 ;\           
 -0.2013 , -0.0009 ;\      
 -0.4020 , -0.0130 ;\      
 -0.5955 , -0.0449 ;\      
 -0.6729 , -0.0720 ;\      
 -0.7433 , -0.0947 ;\      
 -0.8066 , -0.1333 ;\      
 -0.8711 , -0.2084 ;\      
 -0.9074 , -0.2417 ;\      
 -0.9409 , -0.2458 ;\      
 -0.9698 , -0.1784 ;\      
 -0.9801 , -0.1351 ;\      
 -0.9893 , -0.0796 ;\      
 -1.0000 ,  0.0000 ];
 
elseif (Re == 10000) 
  udata = [ \
  1.0000 ,  1.0000 ;\
  0.9862 ,  0.4916 ;\
  0.9795 ,  0.2775 ;\
  0.9567 ,  0.1649 ;\      
  0.9353 ,  0.0904 ;\      
  0.9051 ,  0.0722 ;\      
  0.8736 ,  0.0681 ;\      
  0.8046 ,  0.0627 ;\      
  0.7227 ,  0.0545 ;\      
  0.6798 ,  0.0434 ;\ 
  0.6066 ,  0.0338 ;\      
  0.4027 ,  0.0246 ;\      
  0.2003 ,  0.0080 ;\     
  0.0000 ,  0.0000 ;\           
 -0.2031 , -0.0124 ;\      
 -0.4013 , -0.0275 ;\      
 -0.6021 , -0.0366 ;\      
 -0.6798 , -0.0406 ;\      
 -0.7429 , -0.0544 ;\      
 -0.8060 , -0.1007 ;\      
 -0.8749 , -0.1385 ;\      
 -0.9064 , -0.1750 ;\      
 -0.9379 , -0.2027 ;\      
 -0.9638 , -0.1731 ;\      
 -0.9826 , -0.1353 ;\      
 -0.9910 , -0.0606 ;\      
 -1.0000 ,  0.0000 ];

end

y_CCE = udata(:,1);
y_CCE = (1 + y_CCE)/2;
uvelE = udata(:,2);

if (Re == 3200)
vdata = [ \
  1.0000 ,  0.0000 ;\
  0.9888 , -0.0861 ;\
  0.9747 , -0.1521 ;\
  0.9621 , -0.2310 ;\
  0.9354 , -0.3515 ;\
  0.9045 , -0.4161 ;\
  0.8679 , -0.3572 ;\
  0.8006 , -0.1994 ;\
  0.7346 , -0.0890 ;\
  0.6671 , -0.0603 ;\
  0.6025 , -0.0531 ;\
  0.4003 , -0.0172 ;\
  0.2008 ,  0.0000 ;\
  0.0000 ,  0.0004 ;\
 -0.1980 ,  0.0292 ;\
 -0.3975 ,  0.0574 ;\
 -0.6025 ,  0.1406 ;\
 -0.6657 ,  0.1764 ;\
 -0.7330 ,  0.1879 ;\
 -0.7992 ,  0.2101 ;\
 -0.8638 ,  0.2050 ;\
 -0.9059 ,  0.1944 ;\
 -0.9321 ,  0.1793 ;\
 -0.9579 ,  0.1479 ;\
 -0.9719 ,  0.0990 ;\
 -0.9846 ,  0.0689 ;\
 -1.0000 ,  0.0000 ];
 
elseif (Re == 7500)
  vdata = [ \
  1.0000 ,  0.0000 ;\
  0.9872 , -0.1292 ;\
  0.9768 , -0.2436 ;\
  0.9607 , -0.3466 ;\
  0.9308 , -0.4241 ;\
  0.8952 , -0.3233 ;\
  0.8569 , -0.1844 ;\
  0.8113 , -0.0964 ;\
  0.7486 , -0.0491 ;\
  0.6200 , -0.0266 ;\
  0.4202 , -0.0119 ;\
  0.0000 ,  0.0005 ;\
 -0.1991 ,  0.0196 ;\
 -0.4019 ,  0.0386 ;\
 -0.5990 ,  0.0659 ;\
 -0.6759 ,  0.0808 ;\
 -0.7416 ,  0.0955 ;\
 -0.8003 ,  0.1130 ;\
 -0.8700 ,  0.1434 ;\
 -0.9091 ,  0.1451 ;\
 -0.9398 ,  0.1203 ;\
 -0.9722 ,  0.0881 ;\
 -0.9865 ,  0.0416 ;\
 -1.0000 ,  0.0000 ];
 
 elseif (Re == 10000)
  vdata = [ \
  1.0000 ,  0.0000 ;\
  0.9872 , -0.1464 ;\
  0.9722 , -0.2640 ;\
  0.9667 , -0.3395 ;\
  0.9401 , -0.3989 ;\
  0.9156 , -0.3115 ;\
  0.8892 , -0.2053 ;\
  0.8426 , -0.0877 ;\
  0.8008 , -0.0577 ;\
  0.7389 , -0.0420 ;\
  0.6783 , -0.0349 ;\
  0.5979 , -0.0235 ;\
  0.4008 , -0.0185 ;\
  0.1993 , -0.0010 ;\
  0.0000 ,  0.0266 ;\
 -0.2008 ,  0.0326 ;\
 -0.4036 ,  0.0329 ;\
 -0.6021 ,  0.0583 ;\
 -0.6839 ,  0.0700 ;\
 -0.7417 ,  0.0887 ;\
 -0.8037 ,  0.1004 ;\
 -0.8713 ,  0.1134 ;\
 -0.9066 ,  0.1264 ;\
 -0.9348 ,  0.1078 ;\
 -0.9615 ,  0.0710 ;\
 -0.9797 ,  0.0481 ;\
 -1.0000 ,  0.0000 ]; 
 
end

x_CCE = vdata(:,1);
x_CCE = (1 + x_CCE)/2;
vvelE = vdata(:,2);

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
  
end
