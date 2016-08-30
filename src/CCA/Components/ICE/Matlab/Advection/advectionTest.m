
%______________________________________________________________________
% advectionTest.m               10/06
% This script tests the advection operator used in ICE.  
% The mass and internal energy are being advected.
%
% There is support for a non-uniform mesh. 
%
% reference:  "CompatibleFluxes for van Leer Advection", VanderHeyden,
%             Kashiwa, JCP, 146, 1-28, 1998
clear all;
close all;
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%______________________________________________________________________
%     Problem Setup
desc     = {'Matlab script:2nd Order Advection operator','Refinement ratio:2',' Backward differencing at the CFI',' Gradient limiter hardcoded to 0 at CFI'};
nCells   = 100;               % number of cells
CFL      = 0.75;         
velocity = 3.125;             % uniform velocity
density  = 1.0;
delT     = 1000;
nTimesteps  = 100;
dx_initial  = 0.01;          % initial value for dx

advOrder  = 2;               % first order = 1, second order = 2

refineRatio = 2;
fl_begin    = 0.3;           % fine level begin
fl_end      = 0.5;           % fine level end
%_______________________________
gradLim     =[0:nCells+1];   %gradient Limiter 
grad_x      =[0:nCells+1];   %gradient

x           =[0:nCells+1];   % CC position
dx          =[0:nCells+1];   % cell length
q           =[0:nCells+1];  
q_advected  =[0:nCells+1];   % advected value
xvel_FC     =[0:nCells+1];   % face-centered vel (u)
mass        =[0:nCells+1];   % mass
mass_L      =[0:nCells+1];   % mass (Lagrangian)
int_eng_L   =[0:nCells+1];   % Internal Energy (Lagrangian)
temp        =[0:nCells+1];   % Temperature
mass_slab   =[0:nCells+1];   % mass in slab
mass_vrtx_1 =[0:nCells+1];   % mass at vertex
mass_vrtx_2 =[0:nCells+1];   % --------//-------

%__________________________________
%     Initialization    
x(1)  = 0;
dx(1) = dx_initial;

for(j = 2:nCells+1 )
  dx(j) = dx_initial;
  
  % fine level
  if(x(j-1) > fl_begin) && (x(j-1) < fl_end) 
    dx(j) = dx_initial/refineRatio;
  end
  x(j) = x(j-1) + dx(j);
end

% Top hat distribution for mass 
% and temperature
for(j = 1:nCells+1 )
  xvel_FC(j) = velocity;
  mass(j)  = density * dx(j);
  temp(j)   = 0.0;
  
  if (j >20) & (j < 30)
    mass(j)  = (density * dx(j));
    temp(j)   = 1.0;
  end
end

%__________________________________
% sum the conserved quantities
junk = [0:nCells+1];
sum_int_engInitial = sum(junk(1:nCells));
sumMassInitial     = sum(mass(1:nCells));
fprintf('Initial conditions: sum Mass %e', sumMassInitial);

%__________________________________
% aggressive timestepping
for(j = 1:nCells )
  delT = min(delT,(CFL* dx(j)/velocity));
end

delT
%______________________________________________________________________
%     Time integration loop
for( t = 1:nTimesteps)
  fprintf('\n___________________________________________%i\n', t);
 
  %__________________________________
  % Compute Lagrangian Values
  for(j = 1:nCells+1 )
    mass_L(j)    = mass(j);
    temp_L(j)    = temp(j);
    int_eng_L(j) = mass_L(j) * temp_L(j);
  end
  sumMass(t) = sum(mass_L(1:nCells))/sumMassInitial;
  sumMass(t)
  sumIntEng(t) = sum(int_eng_L(1:nCells)); 
  %__________________________________
  % Advect and advance in time 
  % compute the outflux volumes
  [ofs, rx] = OutFluxVol(xvel_FC, delT, dx, nCells);
  
  %__________________________________
  %  M A S S  
  % uses van Leer limiter
  fprintf ('mass \n');
  [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = advectMass(mass_L, ofs, rx, xvel_FC, dx, nCells, advOrder);  
  
  for(j = 1:nCells-1 )
    mass(j) = mass_L(j) + q_advected(j);
  end
  
  % plot results
  subplot(4,1,1), plot(x(1:nCells),mass(1:nCells), '-+r')
  xlim([0 x(nCells)]);
  legend('mass');
  tit = sprintf('CFL %i',CFL);
  title([desc(1),desc(2),desc(3),desc(4)]);
  grid on;
 
  subplot(4,1,2), plot(x(1:nCells-1),gradLim(1:nCells-1), '-+r')
  xlim([0 x(nCells)]);
  legend('gradient Limiter mass')
  grid on; 

  %__________________________________
  %  I N T E R N A L   E N E R G Y
  % uses compatible flux limiter
  fprintf ('InternalEnergy \n');
  [q_advected,gradLim,grad_x] = advectQ(int_eng_L, mass_L, mass_slab,mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, dx, nCells,advOrder );
  
  for(j = 1:nCells-1 )
    temp(j) = (int_eng_L(j) + q_advected(j))/(mass(j) + 1e-100);
  end
  
  % plot results
  subplot(4,1,3), plot(x(1:nCells),temp(1:nCells),'-+b')
  xlim([0 x(nCells)]);
  legend('Temperature');
  grid on;
  
  subplot(4,1,4), plot(x(1:nCells-1),gradLim(1:nCells-1),'-+b')
  xlim([0 x(nCells)]);
  legend('gradient Limiter Temperature');
  grid on;

  M(t) = getframe(gcf);
end

%______________
% plot the conserved sumMass sumInternal Energy
 figure(2)
 subplot(2,1,1),plot(sumMass);
 legend('sumMass');
 subplot(2,1,2),plot(sumIntEng);
 legend('sumIntEng');
%ylim( [(1.0 -1e-15) (1.0 + 1e-15)] )

%______________
% Movie making
%hFig = figure;
%movie(hFig,M,2,10)
%movie2avi(M,'test.avi','fps',5,'quality',100);