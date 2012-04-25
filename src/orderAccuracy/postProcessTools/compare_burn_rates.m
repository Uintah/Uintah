#! /usr/bin/octave -qf
%_________________________________
% This octave file plots the burn rate vs the average pressure of a 1D strand burner simulation
% and compares the results to experimental data
% Experimental data was run at 298K
%
% Need to have totalMassBurned, press_CC, vol_frac, and modelMass_src as saved tags in your input file
% Make sure puda file has jacquie.cc file 
%
%  Example usage:
%  compare_BurnRates.m -pDir 1 -mat 1 -SA 0.001 -rho_CC 1832 -intTemp 298  
%_________________________________
clear all;
close all;
format short e;

function Usage
  printf('compare_BurnRates.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName> - name of the uda file \n')                                                  
  printf('  -pDir <1,2,3>       - principal direction \n')                                                   
  printf('  -mat                - material index \n')  
  printf('  -CS                 - cell size \n')                                                       
  printf('  -ts                 - Timestep to compute L2 error, default is the last timestep\n') 
  printf('  -o <fname>          - Dump the output (L2Error) to a file\n')   
  printf('  -rho_CC             - density of explosive material\n')  
  printf('  -intTemp            - initial temp of explosive material (can only be 298, 373 or 423 K) \n')                               
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
ts          = 999;
rho_CC      = 1832;
intTemp     = 298;

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
  elseif (strcmp(option,"-rho_CC") )
    rho_CC = str2num(opt_value);
  elseif (strcmp(option,"-intTemp") )
    intTemp = str2num(opt_value);
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
%  extract initial conditions and grid information from the uda file
c0 = sprintf('puda -gridstats %s >& tmp',uda); unix(c0);

[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp         |cut -d":" -f2 | tr -d "[]"');

resolution   = str2num(r1);
domainLength = str2num(r2);
deltaX = domainLength./resolution;
dx = -9;
SA = -9;

% find the direction burning is occuring 
if(pDir == 1)
  dx   = deltaX(1)
  dy   = deltaX(2);
  SA   = dx * dy;
  yDir = 2;
elseif(pDir == 2)   
  dx = deltaX(2)
  yDir = -9;
elseif(pDir == 3)
  dx = deltaX(3)
  yDir = -9;
end

%__________________________________
%compute the exact solution 
if(pDir == 1)
  startEnd = sprintf('-istart 0 0 0 -iend %i 0 0',resolution(pDir)-1);
elseif(pDir == 2)
  startEnd = sprintf('-istart 0 0 0 -iend 0 %i 0',resolution(pDir)-1);
elseif(pDir == 3)
  startEnd = sprintf('-istart 0 0 0 -iend 0 0 %i',resolution(pDir)-1);
end


%______________________________________________________________________
%  Plot TotalMassBurned
% import the data into arrays
c1          = sprintf('chmod go+rwX %s.000',uda);
[s1, r1]    = unix(c1);
c2          = sprintf('cp %s.000/totalMassBurned.dat .',uda);
[s2, r2]    = unix(c2);

tmp         = load('totalMassBurned.dat');
massBurned = tmp(:,2);
time_mbr   = tmp(:,1);
nLen_mbr = length(massBurned);

%______________________________________________________________________
% V E R I F I C A T I O N  
%for c=1:nLen_mbr
 %  massBurned(c) = c;
%end

massBurned
time_mbr
%______________________________________________________________________

%  You may want to smooth the mass burned data,  See 
%  http://octave.sourceforge.net/data-smoothing/function/regdatasmooth.html
%  for an example.

%__________________________________
% compute gradient of mass burned

massBurnedRate = zeros(nLen_mbr,1);

massBurnedRate(1) =0;   % first data point
for c=1:nLen_mbr-1
 dt                = time_mbr(c+1) - time_mbr(c);
 dmb               = massBurned(c); %totalMassBurned is mass burned at that timestep
 massBurnedRate(c+1) = dmb/dt;


end  

massBurnedRate
length(time_mbr);
length(massBurnedRate);

%__________________________________
%  Plot it up
subplot(2,1,1)
  plot(time_mbr, massBurned, 'b:o;;')
  xlabel('Time (s)')
  ylabel('Burned Mass (kg)')
  title('Mass Burned vs Time');
  grid on;

subplot(2,1,2)
  plot(time_mbr, massBurnedRate, 'b:o;;')
  xlabel('Time (s)')
  ylabel('MassBurnRate (kg/s)')
  title('Mass Burn Rate vs Time');
  grid on;
  
  unix('/bin/rm massBurn.ps >&/dev/null');
  print('massBurned.ps','-dps', '-FTimes-Roman:14');

% move data to uda
  c3 = sprintf('mv massBurned.ps %s.000',uda);
  unix(c3);
 
%______________________________________________________________________
% compute the average pressure inside of the region with gas at every timestep
c4        = sprintf('puda -jacquie -matl 1 %s >& /dev/null', uda);
[s4, r4]  = unix(c4);

if( s4 ~=0  )
  disp('warning: you are not using the modified puda with jacquie changes.');
end

tmp       = load('AverageBurnRate.dat');
avePress  = tmp(:,2); 
time_ap   = tmp(:,1);     % physical time when average pressure was computed

% move data to uda
c5        = sprintf('mv AverageBurnRate.dat %s.000',uda);
unix(c5);


% find the mass burn rate (mbr_clipped) at the same physical time as the average pressure
nLen_ap = length(avePress);
mbr_clipped = zeros(nLen_ap,1);

for i=1:nLen_ap
  t = time_ap(i);
  
  for j=1:nLen_mbr
    if ( abs(time_mbr(j).-t)/t < 1e-10)
      c = j;
    end
  end
 
  try
    mbr_clipped(i) = massBurnedRate(c);
  catch
    disp ( 'Warning:  could not find physical time in mass Burned Rate time array');
    t
  end_try_catch
end

mbr_clipped
avePress
time_ap
%__________________________________
% Load and copy experimental data for specified inital temp
% If inial temp is not specified 298 is assumed
if(intTemp == 298)
  tmp = load('ExpData.dat');
  avePress_exp = tmp(:,1);  
  burnVel_exp = tmp(:,2);
elseif(intTemp == 423)   
  tmp = load('Exp423.dat');
  avePress_exp = tmp(:,1);  
  burnVel_exp = tmp(:,2);
elseif(intTemp == 373)
  tmp = load('Exp373.dat');
  avePress_exp = tmp(:,1);  
  burnVel_exp = tmp(:,2);
end


%______________________________________________________________________
% Determine mass burned rate in m/s
burnVel = abs(mbr_clipped./(rho_CC * SA));

% Increase X and Y range to ensure all data points are plotted
maxBurnVel = max(burnVel) + 1e-3;
maxPress   = max(avePress) + 1e5;

%  Plot mass burned rate versus average pressure
figure(2)
subplot(2,1,1)
 plot(avePress, burnVel, 'bo;Computed;', avePress_exp, burnVel_exp, 'r+;Experimental;')
 xlabel('Average Pressure(Pa)')
 ylabel('Mass Burn Rate (m/s)')
 axis([0,maxPress,0,maxBurnVel])
 legend({"Computed","Experimental"},"location","northwest")
 title('Mass Burn Rate vs Average Pressure');
 grid on;

subplot(2,1,2)
 loglog(avePress, burnVel, 'bo;computed;', avePress_exp, burnVel_exp, 'r+;Experimental;')
 xlabel('Average Pressure(Pa)')
 ylabel('Mass Burn Rate (m/s)')
 legend({"Computed","Experimental"},"location","northwest")
 title('Mass Burn Rate vs Average Pressure');
 grid on;

 unix('/bin/rm mbr_ap.ps >&/dev/null');
 print('mbr_ap.ps','-dps', '-FTimes-Roman:14');
  

% move plot to uda file  
  c7 = sprintf('mv mbr_ap.ps %s.000',uda);
  unix(c7);
 
% make copy of plot with uda name for easier access   
  c8          = sprintf('cp %s.000/mbr_ap.ps %s_mbr_ap.ps',uda,uda);
  [s8, r8]    = unix(c8);

%Plot average pressure vs time 
figure(3)
 plot(time_ap, avePress, 'b:o;;')
 ylabel('Average Pressure(Pa)')
 xlabel('Time (s)')
 title('Average Pressure vs Time');
 grid on;

unix('/bin/rm Time_ap.ps >&/dev/null');
  print('Time_ap.ps','-dps', '-FTimes-Roman:14');

% move plot to uda file  
  c9 = sprintf('mv Time_ap.ps %s.000',uda);
  unix(c9);
  
%______________________________________________________________________
% dump the data to a file for gnuplot script

fid = fopen('AP_mbr.dat', 'a');
for i=1:nLen_ap
  fprintf(fid," %16.15f %16.15f\n", avePress(i), burnVel(i));
end

fclose(fid);
 c10 = sprintf('mv AP_mbr.dat %s.000',uda);
  unix(c10);


  
