%______________________________________________________________________
% advectionTest.m               12/04
% This script tests the algoritm for the advection operator used in ICE.  
% The density and internal energy are being advected.
%
% Velocity:       u = 1.0 (Uniform)
% Density:       inverse top hat distribution
% passiveScalar: top hat distribution.
%
% reference:  "CompatibleFluxes for van Leer Advection", VanderHeyden,
%             Kashiwa, JCP, 146, 1-28, 1998
clear all;
close all;
globalParams;                                       % Load global parameters

setenv('LD_LIBRARY_PATH', ['/usr/lib']);
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%______________________________________________________________________
%     Problem Setup

% Set Parameters
CFL         = 0.999;                                 % sigma
velocity    = 1.0;                                   % Uniform velocity (u)

% Geometry
P.boxLower          = 0;                            % Location of lower-left corner of domain
P.boxUpper          = 1;                            % Location of upper-right corner of domain

% Grid
P.nCells            =100;                           % Number of cells in each direction
P.extraCells        = 1;                            % Number of ghost cells in each direction


%================ Grid Struct (G) ======= ================
G.nCells            = P.nCells ;                    % # interior cells
G.delX              = 1;                            % Cell length
G.ghost_Left        = 1;                            % Index of left ghost cell
G.ghost_Right       = G.nCells+2*P.extraCells;      % Index of right ghost cell
G.first_CC          = 2;                            % Index of first interior cell
G.first_FC          = 2;                            % Index of first interior xminus face
G.last_CC           = G.nCells+1;                   % Index of last interior cell
G.last_FC           = G.ghost_Right;                % index of last xminus face

%================ ICE Interal Parameters, Debugging Flags ================
% Debug flags
P.compareUintah     = 0;                            % Compares vs. Uintah ICE and plots results
P.debugSteps        = 0;                            % Debug printout of steps (tasks) within timestep
P.debugAdvectRho    = 0;                            % Debug printouts in advectRho()
P.debugAdvectQ      = 0;                            % Debug printouts in advectQ()
P.printRange        = [48:56];                      % Range of indices to be printed out (around the shock front at the first timestep, for testing)
P.plotInitialData   = 1;                            % Plots initial data
P.advectionOrder    = 1;                            % 1=1st-order advection operator; 2=possibly-limited-2nd-order
P.writeData         = 1;
%______________________________________________________________________
%     Allocate arrays

totCells        = P.nCells + 2*P.extraCells;              % Array size for CC vars
x_CC            = zeros(G.ghost_Left,G.ghost_Right);      % Cell centers locations (x-component)
mass_CC         = zeros(G.ghost_Left,G.ghost_Right);      % Density mass
xvel_CC         = zeros(G.ghost_Left,G.ghost_Right);      % Velocity u (x-component)
f_CC            = zeros(G.ghost_Left,G.ghost_Right);      % Pressure increment due to dilatation

mass_L_CC       = zeros(G.ghost_Left,G.ghost_Right);      % Mass ( = mass * cell volume )
f_L_CC          = zeros(G.ghost_Left,G.ghost_Right);      % Momentum ( = mass * velocity )

x_FC            = zeros(G.ghost_Left,G.last_FC);          % Face centers locations (x-component)  
xvel_FC         = zeros(G.ghost_Left,G.last_FC);          % Velocity u (x-component)              

% Node Centered (NC) 
totNodes        = P.nCells + 1;                           % Array size for NC vars
mass_vrtx_1     = zeros(1,totNodes);                      % Mass at vertices (for advection)
mass_vrtx_2     = zeros(1,totNodes);                      % --------//-------  (for advection)


delT        = CFL * G.delX /velocity;                     % From sigma = u*delT/delX

%______________________________________________________________________
%     Initialization    
d_SMALL_NUM = 1e-100;                                     % A small number (for bullet-proofing
d_TINY_RHO  = 1.0e-12

x_CC    = ([G.ghost_Left:G.ghost_Right]-1-0.5).*G.delX; % Cell centers coordinates (note the "1-based" matlab index array)

x_FC(1) = -G.delX;
for j = G.first_FC:G.last_FC   % Loop over all xminus cell faces
  x_FC(j) = (j-G.first_FC).*G.delX;  
end

figure(1);
for j = G.ghost_Left:G.ghost_Right
  xvel_CC(j)    = velocity;
  xvel_FC(j)    = velocity;    
  mass_CC(j)    = 0.5;      
     
  f_CC(j)       = 0.0; 
          
  if ((j > 10) && (j < 30))       % top-hat
    mass_CC(j)   = 0.001;     
    f_CC(j)     = 1.0;       
  end
end
xvel_FC(G.nCells+1) = velocity;

 plot(x_CC, f_CC);
input('hit return')
%______________________________________________________________________
%     Time integration loop
for t = 1:25
    fprintf('\n___________________________________________%i\n', t);
    
    %__________________________________
    % Compute Lagrangian Values
    mass_L_CC  = mass_CC;                      
    f_L_CC     = mass_L_CC .* f_CC;
    
    %__________________________________
    % Advect and advance in time 
    % compute the outflux volumes
    [ofs, rx] = OutFluxVol(xvel_CC, xvel_FC, delT, G);
    
    %__________________________________
    % D E N S I T Y  
    % Uses van Leer limiter
    fprintf ('density \n');
    [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ...
        advectRho(mass_L_CC, ofs, rx, xvel_FC, G);  
    

    mass_CC = mass_L_CC + q_advected;
    
    % Plot mass results
    subplot(4,1,1), plot(x_CC, mass_CC, '-r');
    xlim([0 100]);
    legend('mass');
    grid on;
    
    subplot(4,1,2), plot(x_CC, gradLim, '-r');
    xlim([0 100]);
    legend('gradLim mass')
    grid on;
    
    %__________________________________
    %  P A S S I V E   S C A L A R
    % Uses compatible flux limiter
    fprintf ('passive scalar \n');
    
    [q_advected, gradLim, grad_x] = ...
        advectQ(f_L_CC, mass_L_CC, mass_slab, mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, G);
    
    f_CC = (f_L_CC + q_advected) ./ (mass_CC + d_SMALL_NUM);
    
    % Plot f results
    subplot(4,1,3), plot(x_CC, f_CC);
    xlim([0 100]);
    legend('f');
    
    subplot(4,1,4), plot(x_CC, gradLim);
    xlim([0 100]);
    legend('gradLim f');
    grid on;
    
    M(t) = getframe(gcf);
end

%hFig = figure(2);
%movie(hFig,M,1,10)
