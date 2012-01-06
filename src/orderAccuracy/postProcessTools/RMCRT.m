#! /usr/bin/env octaveWrap
%_________________________________
% This octave file plots the radiative flux divergence (deldotq)
% of a 2D center slice of a 3D domainprofile and computes
% the L2norm of the  divergence of q
% 
%
% Example usage:
%_________________________________
clear all;
close all;
format short e;

%______________________________________________________________________
%______________________________________________________________________
%   B E N C H M A R K   1
function [divQ_exact] = benchMark1(x_CC)
  printf("BenchMark 1");
  
  x_exact = 0:(1/40):1;
  ExactSol = zeros(41);

  ExactSol = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 3.07561, 2.94305, 2.81667, 2.69362,...
  2.57196, 2.45069, 2.32896, 2.20616, 2.08181,...
  1.95549, 1.82691, 1.69575, 1.56180, 1.42485, 1.28473, 1.1413,...
  .99443, .84405, .6901, .53255, .37165];

  % The makes the exact solution symmetric
  j=41;
  for q=1:20
    ExactSol(q) = ExactSol(j);
    j=j-1;
  end

  %Do a pchip interpolation to any resolution
  divQ_exact = interp1(x_exact, ExactSol, x_CC, 'pchip');
  
endfunction

%______________________________________________________________________
%    B E N C H M A R K   2    
function [DivQ] = benchMark2(x_CC)
  printf("BenchMark 2\n");

  [s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
  resolution = str2num(r1);
  zDir = 3;
  n = resolution(zDir); %number of cells in the z direction (between the plates)

  %This script computes the exact solution to any resolution at the cell centers 
  %for the benchmark case described by Modest in section 13.5.
  %Plane medium with specified temp field.  We fix the abskg at 1/m
  %and fix the T of the two walls to be the same.  No scattering.  Analytical  soln
  %to DivQ is given by the last eqn of Modest  section 13.5.  

  eps = 1;        % Emissivity of walls
  Tw = 1000;      % Temp of walls
  Tm = 1500;      % Temp of mediumm
  sigma = 5.67051e-8;
  k = 1;          %optical thickness at x=L
  %the below line is adjusted to give cell centered answers
  x = linspace(1/(2*n),k*(1-1/(2*n)), n*k); 

  theta = linspace(0,pi/2, 100000); %use 100,000 to get 9 digits of accuracy
  mu = cos(theta);


  %compute ExpInt2 and ExpInt3 as functions of Tau
  ExpInt2 = zeros(length(x),1);%Exponential integrals from Modest pg 429
  ExpInt3 = zeros(length(x),1);

  %perfrom the integration over dmu using trapezoid  rule
  for ix=1:length(ExpInt2)
    for imu = 2:length(mu)
      ExpInt2(ix) =  ExpInt2(ix) + ( exp(-x(ix)/mu(imu-1)) + exp(-x(ix)/mu(imu)) )/2 * (mu(imu-1) - mu(imu)); %final term is dmu
      ExpInt3(ix) =  ExpInt3(ix) + ( mu(imu-1) * exp(-x(ix)/mu(imu-1)) + mu(imu) * exp(-x(ix)/mu(imu)))/2 * (mu(imu-1) - mu(imu)); 
    end
  end

  %compute the exact soln of DivQ according to final eqn in Modest section 13.5
  DivQ = zeros(n,1);%radiative flux divergence
  for ix=1:length(DivQ)-1
    DivQ(ix) = sigma*(Tw^4 - Tm^4) * -2*( ExpInt2(ix*k) + ExpInt2(length(DivQ)*k-ix*k +k) ) / (  1 +  (1/eps -1)*( 1-2*ExpInt3(length(DivQ)*k) )  ) ;
  end
  ExpInt2_0 = 1;%Definition of ExpInt at Tau=0;

  DivQ(length(DivQ)) = sigma*(Tw^4 - Tm^4) * -2*( ExpInt2(length(DivQ)*k) + ExpInt2_0 ) / (  1 +  (1/eps -1)*( 1-2*ExpInt3(length(DivQ)*k) )  ) ;
  DivQ(length(DivQ)) = ( DivQ(length(DivQ)) + DivQ(length(DivQ)-1) ) / 2; %This is how we get the cell centered value of the last point. 
  %We use the boundary condition (two lines up) averaged with the previous

  printf("...end\n");
endfunction


%______________________________________________________________________
%   B E N C H M A R K   3
function [ExactSol] = benchMark3(x_CC)
  printf("BenchMark 3");
  
  load bench3Exact.txt;

  [s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
  resolution = str2num(r1);

  zDir = 3;
  n = resolution(zDir); %number of cells in the z direction (between the plates)

  ExactSol = bench3Exact;
  
  x243 = 1/(2*243):1/243 :1-1/(2*243);

  Exact81 = ExactSol(2:3:242);
  Exact27 = Exact81(2:3:80);
  Exact9  = Exact27(2:3:26);
  Exact3  = Exact9(2:3:8);

  if (n == 81)
    ExactSol = Exact81;
  end
  if (n == 27)
    ExactSol = Exact27; 
  end
  if (n == 9)   
    ExactSol = Exact9;
  end
  if (n == 3)   
    ExactSol = Exact3;
  end
  if (n !=3 && n != 9 && n!= 27 && n != 81 && n!=243)
    ExactSol = interp1(x243, ExactSol, x_CC, 'pchip');
  end

endfunction

%______________________________________________________________________
%   B E N C H M A R K   4
function [ExactSol] = benchMark4(x_CC)
  printf("BenchMark 4");
  
  load b4divQCenterExact.txt;

  [s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp |cut -d":" -f2 | tr -d "[]int"');
  resolution = str2num(r1);

  zDir = 3;
  n = resolution(zDir); %number of cells in the z direction (between the plates)

  ExactSol = b4divQCenterExact;
  
  x243 = 1/(2*243):1/243 :1-1/(2*243);

  Exact81 = ExactSol(2:3:242);
  Exact27 = Exact81(2:3:80);
  Exact9  = Exact27(2:3:26);
  Exact3  = Exact9(2:3:8);

  if (n == 81)
    ExactSol = Exact81;
  end
  if (n == 27)
    ExactSol = Exact27; 
  end
  if (n == 9)   
    ExactSol = Exact9;
  end
  if (n == 3)   
    ExactSol = Exact3;
  end
  if (n !=3 && n != 9 && n!= 27 && n != 81 && n!=243)
    ExactSol = interp1(x243, ExactSol, x_CC, 'pchip');
  end

endfunction

%______________________________________________________________________


function Usage
  printf('RMCRT.m <options>\n') 
  printf('options:\n') 
  printf(' -uda <udaFileName>  - name of the uda file \n') 
  printf(' -bm  <1,2,..>       - benchmark test to compare against \n')
  printf(' -pDir <1,2,3>       - principal direction \n') 
  printf(' -mat                - material index \n') 
  printf(' -L                  - level index, default 0\n')
  printf(' -plot <true, false> - produce a plot \n') 
  printf(' -ts                 - Timestep to compute L2 error, default is the last timestep\n') 
  printf(' -o <fname>          - Dump the output (L2norm) to a file\n') 
  printf('----------------------------------------------------------\n')
end 


%______________________________________________________________________
%______________________________________________________________________

%________________________________ 
% Parse User inputs 
nargin = length(argv);
if (nargin == 0)
  Usage
exit
endif

%__________________________________
% defaults
pDir        = 1;
mat         = 0;
makePlot    = "true";
ts          = 999;
output_file = 'L2norm';
level       = 0;
benchmark   = -9;

% Parse the command line arguments
arg_list = argv ();
for i = 1:2:nargin
  option = sprintf("%s",arg_list{i} );
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
  elseif (strcmp(option,"-L") ) 
    level = str2num(opt_value);
  elseif (strcmp(option,"-bm") ) 
    benchmark = str2num(opt_value);
  end 
end

%__________________________________
% bulletproofing
if( benchmark == -9 )
  error ("An invalid benchmark test was selected.  Please correct this");
  exit
end

%________________________________
% do the Uintah utilities exist
[s0, r0]=unix('puda >& /dev/null');
[s1, r1]=unix('lineextract >& /dev/null');

if( s0 ~=0 || s1 ~= 0 )
  disp('Cannot execute uintah utilites puda, lineextract');
  disp(' a) make sure you are in the right directory, and');
  disp(' b) the utilities (puda/lineextract) have been compiled');
end


%________________________________
% extract the physical time
c0 = sprintf('puda -timesteps %s | grep : | cut -f 2 -d":" >& tmp',uda);
[status0, result0]=unix(c0);
physicalTime = load('tmp');

if(ts == 999) % default
  ts = length(physicalTime);
end
%________________________________
% extract the grid information from the uda file
% at the right level
c0 = sprintf('puda -gridstats %s >& tmp',uda); unix(c0);
c1 = sprintf('sed -n /"Level: index %i"/,/"Total Number of Cells"/{p} tmp >& tmp.clean',level);
unix(c1);

[s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp.clean |cut -d":" -f2 | tr -d "[]int"');
[s,r2] = unix('grep -m1 -w "Domain Length" tmp |cut -d":" -f2 | tr -d "[]"');

resolution = str2num(r1);
domainLength = str2num(r2);

%__________________________________
% Extract the data from the uda
% find the y direction
xDir = 1;
yDir = 2;
zDir = 3;

xHalf = resolution(xDir)/2.0;
yHalf = resolution(yDir)/2.0;
zHalf = resolution(zDir)/2.0;

if(pDir == 1)
  startEnd = sprintf('-istart %i   %i   %i  -iend   %i   %i  %i',0,     yHalf,  zHalf, resolution(xDir)-1, yHalf, zHalf );
elseif(pDir == 2)
  startEnd = sprintf('-istart %i   %i   %i  -iend   %i  %i   %i',xHalf, 0,      zHalf, xHalf, resolution(yDir)-1, zHalf);
elseif(pDir == 3)
  startEnd = sprintf('-istart %i  %i    %i  -iend   %i  %i   %i',xHalf, yHalf,   0,    xHalf, yHalf, resolution(zDir)-1);
end

c1 = sprintf('lineextract -v %s -l %i -cellCoords -timestep %i %s -o divQ.dat -m %i -uda %s >&/dev/null','divQ',level,ts-1,startEnd,mat,uda);
[s1, r1] = unix(c1);

divQ_sim = load('divQ.dat');
x_CC     = divQ_sim(:,pDir);         % This is actually x_CC or y_CC or z_CC

%__________________________________
% compute the exact solution
if (benchmark == 1)
  [divQ_exact] = benchMark1(x_CC);
elseif (benchmark == 2 )
  [divQ_exact] = benchMark2(x_CC);
elseif (benchmark == 3 )
  [divQ_exact] = benchMark3(x_CC);
end

%______________________________
% compute the L2 norm
clear d;
d = 0;
d = abs(divQ_sim(:,4) - divQ_exact);
L2_norm = sqrt( sum(d.^2)/length(x_CC) );

% write L2_norm to a file
nargv = length(output_file);
if (nargv > 0)
  fid = fopen(output_file, 'w');
  fprintf(fid,'%g\n',L2_norm);
  fclose(fid);
end

% cleanup 
unix('/bin/rm divQ.dat tmp tmp.clean');

%______________________________
% Plot the results
if (strcmp(makePlot,"true"))
  subplot(2,1,1),plot(x_CC, divQ_sim(:,4), 'b:o;computed;', x_CC, divQ_exact, 'r:+;exact;');
  ylabel('divQ');
  xlabel('X');
  title('divQ versus Exact solns');
  grid on;

  subplot(2,1,2),plot(x_CC,d, 'b:+');
  hold on;
  ylabel('|Deldotq - u_{exact}|'); 
  xlabel('X');
  grid on;

  unix('/bin/rm divQ.ps >&/dev/null');
  print('divQ.ps','-dps', '-FTimes-Roman:14');
  pause

end




