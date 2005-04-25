%______________________________________________________________________
% advectionTest.m               12/04
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

% Set Parameters
nCells      = 100;                      % Number of cells
delX        = 1.0;                      % Cell length
CFL         = 0.999;                    % sigma
velocity    = 1.0;                      % Uniform velocity (u)
cv          = 1.0;                      % Specific heat
delT        = CFL * delX / velocity;    % From sigma = u*delT/delX

% Allocate arrays - Cell Centered (CC)
gradLim     = zeros(1,nCells);          % Gradient Limiter 
grad_x      = zeros(1,nCells);          % Gradient of rho
q           = zeros(1,nCells);          % Quantity to be advected (rho/A=rho*T)
q_advected  = zeros(1,nCells);          % Advected flux of quantity
rho         = zeros(1,nCells);          % Density
temp        = zeros(1,nCells);          % Temperature T
rho_L       = zeros(1,nCells);          % Density (Lagrangian)
int_eng_L   = zeros(1,nCells);          % Internal Energy (Lagrangian)
rho_slab    = zeros(1,nCells);          % Density in slab

% Allocate arrays - Face Centered (FC)
xvel_FC     = zeros(1,nCells+1);        % Face-centered vel (u)

% Allocate arrays - Node (vertex) Centered (NC)
rho_vrtx_1  = zeros(1,nCells+1);        % Density at vertices
rho_vrtx_2  = zeros(1,nCells+1);        % --------//-------

%______________________________________________________________________
%     Initialization    
figure(1);
for j = 1:nCells
    xvel_FC(j)    = velocity;
    rho(j)        = 0.5;
    temp(j)       = 0.0;
    if ((j > 10) & (j < 30))
        rho(j)      = 0.001;
        temp(j)     = 1.0;
    end
end
xvel_FC(nCells+1) = velocity;

%______________________________________________________________________
%     Time integration loop
for t = 1:25
    fprintf('\n___________________________________________%i\n', t);
    
    %__________________________________
    % Compute Lagrangian Values
    for j = 1:nCells
        rho_L(j)     = rho(j);
        temp_L(j)    = temp(j);
        int_eng_L(j) = rho_L(j) * temp_L(j) * cv;
    end
    
    %__________________________________
    % Advect and advance in time 
    % compute the outflux volumes
    [ofs, rx] = OutFluxVol(xvel_FC, delT, delX, nCells);
    
    %__________________________________
    % D E N S I T Y  
    % Uses van Leer limiter
    fprintf ('density \n');
    [q_advected, gradLim, grad_x, rho_slab, rho_vrtx_1, rho_vrtx_2] = ...
        advectRho(rho_L, ofs, rx, xvel_FC, delX, nCells);  
    
    for j = 1:nCells
        rho(j) = rho_L(j) + q_advected(j);
    end
    
    % Plot rho results
    subplot(4,1,1), plot(rho,     '-r');
    xlim([0 100]);
    legend('rho');
    grid on;
    
    subplot(4,1,2), plot(gradLim, '-r');
    xlim([0 100]);
    legend('gradLim rho')
    grid on;
    
    %__________________________________
    % I N T E R N A L   E N E R G Y
    % Uses compatible flux limiter
    fprintf ('InternalEnergy \n');
    [q_advected, gradLim, grad_x] = ...
        advectQ(int_eng_L, rho_L, rho_slab, rho_vrtx_1, rho_vrtx_2, ofs, rx, xvel_FC, delX, nCells);
    
    for j = 1:nCells
        temp(j) = (int_eng_L(j) + q_advected(j))/(rho(j) + 1e-100);
    end
    
    % Plot temp results
    subplot(4,1,3), plot(temp);
    xlim([0 100]);
    legend('temp');
    
    subplot(4,1,4), plot(gradLim);
    xlim([0 100]);
    legend('gradLim int eng');
    grid on;
    
    M(t) = getframe(gcf);
end

%hFig = figure(2);
%movie(hFig,M,1,10)
