
%______________________________________________________________________
% advectionTest.m               11/04
% This script tests the algoritm for the advection operator used in ICE.  
% The density and internal energy are being advected.
%
% Velocity:    u = 1.0 (Uniform)
% Density:     inverse top hat distribution
% Temperature: top hat distribution.
%
% reference:  "CompatibleFluxes for van Leer Advection", VanderHeyden,
%             Kashiwa, JCP, 146, 1-28, 1998
clear all;
close all;
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%______________________________________________________________________
%     Problem Setup
nCells   = 100;             % number of cells
CFL      = 0.75;         
velocity = -3.125;             % uniform velocity
density  = 1.0;
delT     = 1000;
nTimesteps  = 50;
dx_initial  = 0.01;

refineRatio = 2;

gradLim     =[0:nCells+1];    %gradient Limiter 
grad_x      =[0:nCells+1];    %gradient

x           =[0:nCells+1];  % CC position
dx          =[0:nCells+1];  % cell length
q           =[0:nCells+1];  
q_advected  =[0:nCells+1];  % advected value
xvel_FC     =[0:nCells+1];  % face-centered vel (u)
mass        =[0:nCells+1];  % mass
mass_L      =[0:nCells+1];  % mass (Lagrangian)
int_eng_L   =[0:nCells+1];  % Internal Energy (Lagrangian)
temp        =[0:nCells+1];  % Temperature
mass_slab   =[0:nCells+1];  % mass in slab
mass_vrtx_1 =[0:nCells+1];  % mass at vertex
mass_vrtx_2 =[0:nCells+1];  % --------//-------

%__________________________________
%     Initialization    
x(1) = 0;
dx = dx_initial;
for(j = 2:nCells+1 )
  dx(j) = dx_initial;
  
  % fine level
  if(x(j) > 3 & x(j) < 5 )
    dx(j) = dx_initial;
  end
  x(j) = x(j-1) + dx(j);
end

% Top hat distribution for mass 
% linear distribution for temperature
for(j = 1:nCells+1 )
  xvel_FC(j) = velocity;
  mass(j)  = density;
  temp(j)  = j/nCells;
  
  if (j >10) & (j < 30)
    mass(j)  = (density*pi);
  end
end

sumMassInitial = sum(mass(1:nCells));
fprintf('Initial conditions: sum Mass %e', sumMassInitial);


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
  
  %__________________________________
  % Advect and advance in time 
  % compute the outflux volumes
  [ofs, rx] = OutFluxVol(xvel_FC, delT, dx, nCells);
  
  %__________________________________
  %  M A S S  
  % uses van Leer limiter
  fprintf ('mass \n');
  [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = advectMass(mass_L, ofs, rx, xvel_FC, dx, nCells);  
  
  for(j = 1:nCells-1 )
    mass(j) = mass_L(j) + q_advected(j);
  end
  
  % plot results
  subplot(4,1,1), plot(x(1:nCells),mass(1:nCells), '-+r')
  xlim([0 x(nCells)]);
  legend('mass');
  tit = sprintf('CFL %i',CFL);
  title(tit);
  grid on;
  
  subplot(4,1,2), plot(x(1:nCells-1),gradLim(1:nCells-1), '-+r')
  xlim([0 x(nCells)]);
  legend('gradient Limiter mass')
  grid on;
  
  sumMass(t) = sum(mass(1:nCells))/sumMassInitial;
  sumMass(t)
  %__________________________________
  %  I N T E R N A L   E N E R G Y
  % uses compatible flux limiter
  fprintf ('InternalEnergy \n');
  [q_advected,gradLim,grad_x] = advectQ(int_eng_L, mass_L, mass_slab,mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, dx, nCells);
  
  for(j = 1:nCells-1 )
    temp(j) = (int_eng_L(j) + q_advected(j))/(mass(j) + 1e-100);
  end
  
  % plot results
  subplot(4,1,3), plot(x(1:nCells),temp(1:nCells),'-+b')
  xlim([0 x(nCells)]);
  legend('Temperature');
  
  subplot(4,1,4), plot(x(1:nCells-1),gradLim(1:nCells-1),'-+b')
  xlim([0 x(nCells)]);
  legend('gradient Limiter Temperature');
   grid on;
   
  M(t) = getframe(gcf);
end

%______________
% plot the conserved sumMass
figure(2)
plot(sumMass);
ylim( [(1.0 -1e-15) (1.0 + 1e-15)] )

%hFig = figure;
%movie(hFig,M,2,10)
