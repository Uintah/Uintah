
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
delX     = 1.0;             % cell length
CFL      = 0.999;         
velocity = 1.0;             % uniform velocity
delT   = CFL * delX/velocity;


gradLim     =[1:nCells];    %gradient Limiter 
grad_x      =[1:nCells];    %gradient

q           =[0:nCells+1];  
q_advected  =[0:nCells+1];  % advected value
xvel_FC     =[0:nCells+1];  % face-centered vel (u)
rho         =[0:nCells+1];  % Density
rho_L       =[0:nCells+1];  % Density (Lagrangian)
int_eng_L   =[0:nCells+1];  % Internal Energy (Lagrangian)
temp        =[0:nCells+1];  % Temperature
rho_slab   =[0:nCells+1];   % density in slab
rho_vrtx_1 =[0:nCells+1];   % density at vertex
rho_vrtx_2 =[0:nCells+1];   % --------//-------

%__________________________________
%     Initialization    
for(j = 1:nCells+1 )
  xvel_FC(j) = velocity;
  rho(j)  = 0.5;
  temp(j) = 0.0;
  if (j >10) & (j < 30)
    rho(j)  = 0.001;
    temp(j) = 1.0;
  end
end
%______________________________________________________________________
%     Time integration loop
for( t = 1:50)
  fprintf('\n___________________________________________%i\n', t);
  
  %__________________________________
  % Compute Lagrangian Values
  for(j = 1:nCells+1 )
    rho_L(j)     = rho(j);
    temp_L(j)    = temp(j);
    int_eng_L(j) = rho_L(j) * temp_L(j);
  end
  
  %__________________________________
  % Advect and advance in time 
  % compute the outflux volumes
  [ofs, rx] = OutFluxVol(xvel_FC, delT, delX, nCells);
  
  %__________________________________
  %  D E N S I T Y  
  % uses van Leer limiter
  fprintf ('density \n');
  [q_advected, gradLim, grad_x, rho_slab, rho_vrtx_1, rho_vrtx_2] = advectRho(rho_L, ofs, rx, xvel_FC, delX, nCells);  
  
  for(j = 1:nCells-1 )
    rho(j) = rho_L(j) + q_advected(j);
  end
  
  % plot results
  subplot(4,1,1), plot(rho,     '-r')
  xlim([0 100]);
  legend('rho');
  grid on;
  
  subplot(4,1,2), plot(gradLim, '-r')
  xlim([0 100]);
  legend('gradLim rho')
  grid on;
  %__________________________________
  %  I N T E R N A L   E N E R G Y
  % uses compatible flux limiter
  fprintf ('InternalEnergy \n');
  [q_advected,gradLim,grad_x] = advectQ(int_eng_L, rho_L, rho_slab,rho_vrtx_1, rho_vrtx_2, ofs, rx, xvel_FC, delX, nCells);
  
  for(j = 1:nCells-1 )
    temp(j) = (int_eng_L(j) + q_advected(j))/(rho(j) + 1e-100);
  end
  
  % plot results
  subplot(4,1,3), plot(temp)
  xlim([0 100]);
  legend('temp');
  
  subplot(4,1,4), plot(gradLim)
  xlim([0 100]);
  legend('gradLim int eng');
  grid on;
  
  M(t) = getframe(gcf);
end


hFig = figure;
movie(hFig,M,2,10)
