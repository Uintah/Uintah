%ICE ICE Algorithm for the Shock Tube Problem in 1D (main driver) - 12/04
%
% This script tests full ICE time-stepping algorithm for Sod's Shocktube
% problem in one space dimension.
% We solve the compressible Euler equations for density (rho), veclocity
% (u), temperature (T) and pressure (p). Internal energy is also used in
% parts of the timestep (but is linearly related to the temperature).
% 
% Initial conditions: 
%    rho=step function
%    u=0
%    T=300 Kelvin
%    p=constant
%
% reference:  "ICE, explicit pressure, single material, reaction model"
%             Todd Harman, 09/24/04
%             ICE input file: rieman_sm.ups
%   
%   See also SETBOUNDARYCONDITIONS, ADVECTQ.

clear all;
close all;
globalParams;                               % Load global parameters

%______________________________________________________________________
%     Problem Setup

%================ Set Parameters ================

% Geometry
boxLower        = 0;                        % Location of lower-left corner of domain
boxUpper        = 1;                        % Location of upper-right corner of domain

% Grid
nCells          = 100;                      % Number of cells in each direction
extraCells      = 1;                        % Number of ghost cells in each direction

% Time-stepping
maxTime         = 0.0005;                   % Maximum simulation time [sec]
initTime        = 0.0;                      % Initial simulation time [sec]
delt_init       = 1e-6;                     % First timestep [sec]
maxTimeSteps    = 1;                      % Maximum number of timesteps [dimensionless]
CFL             = 0.4;                    % Courant number (~velocity*delT/delX) [dimensionless]

% Material properties (ideal gas)
cv              = 717.5;                    % Specific_heat
gamma           = 1.4;                      % gamma coefficient in the Equation of State (EOS)

%================ ICE Interal Parameters ================


%================ Partition of the domain into regions ================

numRegions      = 2;                        % Partition of the domain into numRegions regions; physical properties are uniform within each region
Region          = cell(numRegions,1);       % This array holds the geometrical partition physical info
count           = 0;

%********** Parameters of the high density left region **********

count           = count+1;
R.label         = 'leftpartition';          % Title of this region
R.min           = 0;                        % Location of lower-left corner of this region [length]
R.max           = 0.5;                      % Location of upper-right corner of this region [length]
R.velocity      = 10.0;                      % Initial velocity
R.temperature   = 300.0;                    % Initial temperature [Kelvin]
R.density       = 1.1768292682926831000;    % Initial density
R.pressure      = 101325.0;                 % Initial pressure (1 atmosphere)
Region{count}   = R;                        % Add region to list

%********** Parameters of the low density right region **********

count           = count+1;
R.min           = 0.5;                      % Location of lower-left corner of this region [length]
R.max           = 1;                        % Location of upper-right corner of this region [length]
R.velocity      = 10.0;                      % Initial velocity
R.temperature   = 300.0;                    % Initial temperature [Kelvin]
%R.density       = 0.11768292682926831000;   % Initial density
%R.pressure      = 10132.50;                 % Initial pressure (0.1 atmosphere)
R.density       = 1.1768292682926831000;   % Initial density
R.pressure      = 101325.0;                 % Initial pressure (0.1 atmosphere)
Region{count}   = R;                        % Add region to list

%______________________________________________________________________
%     Allocate arrays

%================ Cell Centered (CC) ================

totCells    = nCells + 2*extraCells;        % Array size for CC vars
x_CC        = zeros(1,totCells);            % Cell centers locations (x-component)
rho_CC      = zeros(1,totCells);            % Density rho
xvel_CC     = zeros(1,totCells);            % Velocity u (x-component)
temp_CC     = zeros(1,totCells);            % Temperature T
press_CC    = zeros(1,totCells);            % Pressure p
volfrac_CC  = zeros(1,totCells);            % Volume fraction theta, = 1 for this single-material problem
spvol_CC    = zeros(1,totCells);            % Specific volume v0 = theta/rho = 1/(specific density)
mass_CC     = zeros(1,totCells);             % Mass at time n+1 (see step 9)

gradLim     = zeros(1,totCells);            % Gradient Limiter (for advection)
grad_x      = zeros(1,totCells);            % Gradient of rho (for advection)
q_advected  = zeros(1,totCells);            % Advected flux of quantity (for advection)
rho_slab    = zeros(1,totCells);            % Density in slab (for advection)

%================ Cell Centered (CC), Lagrangian values (L) ================

del_mom     = zeros(1,totCells);            % Momentum accumulated sources
del_eng     = zeros(1,totCells);            % Energy accumulated sources
mass_L      = zeros(1,totCells);            % Mass ( = rho * cell volume )
mom_L       = zeros(1,totCells);            % Momentum ( = mass * velocity )
eng_L       = zeros(1,totCells);            % Energy (= mass * internal energy = mass * cv * temperature)


%================ Face Centered (FC) ================

totFaces    = nCells + 1;                   % Array size for FC vars
xvel_FC     = zeros(1,totFaces);            % Velocity u (x-component)
press_FC    = zeros(1,totFaces);            % Pressure p


%================ Node Centered (NC) ================

totNodes    = nCells + 1;                   % Array size for NC vars
rho_vrtx_1  = zeros(1,totNodes);            % Density at vertices (for advection)
rho_vrtx_2  = zeros(1,totNodes);            % --------//-------  (for advection)

%______________________________________________________________________
%     Initialization    

%================ Useful constants ================

d_SMALL_NUM             = 1e-30;                    % A small number (for bullet-proofing)
delX                    = (boxUpper-boxLower) ...   % Cell length
    ./nCells;
delT                    = delt_init;                % Init timestep
ghost_Left              = 1;                        % Index of left ghost cell
ghost_Right             = nCells+2*extraCells;      % Index of right ghost cell
firstCell               = 2;                        % Index of first interior cell
lastCell                = nCells+1;                 % Index of last interior cell

%================ Initialize interior cells ================
% Initial data at t=0 in the interior domain.

x_CC    = ([1:totCells]-1-0.5).*delX;               % Cell centers coordinates (note the "1-based" matlab index array)
for r = 1:numRegions                                % Init each region, assuming they are a non-overlapping all-covering partition of the domain
    R       = Region{r};
    first   = max(firstCell,...
        floor(R.min./delX - 0.5 + firstCell));
    last    = min(lastCell,...
        floor(R.max./delX - 0.5 + firstCell));
    
    for j = first:last    
        rho_CC(j)           = R.density;
        xvel_CC(j)          = R.velocity;
        temp_CC(j)          = R.temperature;
        press_CC(j)         = R.pressure;
    end
    
end

%================ Initialize ghost cells ================
% Impose boundary conditions (determine ghost cell values from 
% interior cell values).

rho_CC      = setBoundaryConditions(rho_CC      ,'rho_CC');
xvel_CC     = setBoundaryConditions(xvel_CC     ,'xvel_CC');
temp_CC     = setBoundaryConditions(temp_CC     ,'temp_CC');
press_CC    = setBoundaryConditions(press_CC    ,'press_CC');

%================ Initialize graphics ================
figure(1);
set(0,'DefaultFigurePosition',[0,0,1024,768]);
%================ Plot results ================

subplot(2,2,1), plot(x_CC,rho_CC);
xlim([boxLower(1) boxUpper(1)]);
legend('\rho');
grid on;

subplot(2,2,2), plot(x_CC,xvel_CC);
xlim([boxLower(1) boxUpper(1)]);
legend('u_1');
grid on;

subplot(2,2,3), plot(x_CC,temp_CC);
xlim([boxLower(1) boxUpper(1)]);
legend('T');
grid on;

subplot(2,2,4), plot(x_CC,press_CC);
xlim([boxLower(1) boxUpper(1)]);
legend('p');
grid on;

%M(tstep) = getframe(gcf);

%______________________________________________________________________
%     Time integration loop
t = initTime + delT;
for tstep = 1:maxTimeSteps
    fprintf('\n_____________________________________tstep=%d, t=%e\n', tstep, t);
    
    %_____________________________________________________
    % 0. Dummy setting for a single-material problem
    % Set the volume fraction and specific volume.
    fprintf('Step 0: dummy setting for single-material\n');

    volfrac_CC      = ones(1,totCells);                             % Single material ==> covers 100% of each cell (volfrac=1)
    spvol_CC        = volfrac_CC ./ (rho_CC + d_SMALL_NUM);         % Specific volume, here also = 1/rho

    
    %_____________________________________________________
    % 1. Compute thremodynamic/transport properties 
    % These are constants in this application and were already computed in the
    % initialization stage.
        
    
    %_____________________________________________________
    % 2. Compute the equilibration pressure 
    % Using an equation of State for ideal gas.
    fprintf('Step 2: compute equilibration pressure\n');

    % Update pressure from EOS
    press_CC        = (gamma-1).*cv.*rho_CC.*temp_CC;               % Evaluate p from EOS
    
    % Compute speed of sound at cell centers
    DpDrho          = (gamma-1).*cv.*temp_CC;                       % d P / d rho
    DpDe            = (gamma-1).*rho_CC;                            % d P / d e
    speedSound_CC   = sqrt(DpDrho + DpDe.*press_CC./ ...
        (rho_CC.^2 + d_SMALL_NUM));                                 % Speed of sound
    
    % Set boundary conditions on p
    press_CC        =  setBoundaryConditions(press_CC,'press_CC');

    
    %_____________________________________________________
    % 3. Compute sources of energy due to chemical reactions
    % Not applicable to this model.

    
    %_____________________________________________________
    % 4. Compute the face-centered velocities
    fprintf('Step 4: compute face-centered velocities\n');

    for j = firstCell:lastCell-1                                  % Loop over faces inside the domain
        left        = j;
        right       = j+1;
        term1       = (...
            rho_CC(left )*xvel_CC(left ) + ...
            rho_CC(right)*xvel_CC(right) ...
            ) / (rho_CC(left) + rho_CC(right) + d_SMALL_NUM);
            
        term2a      = 2.0*spvol_CC(left)*spvol_CC(right) ...
            / (spvol_CC(left) + spvol_CC(right) + d_SMALL_NUM);
            
        term2b      = (press_CC(right) - press_CC(left))/delX;
        
        xvel_FC(j)  = term1 - delT*term2a*term2b;
    end
    
    % Set boundary conditions on u @ face centers
    % Using the B.C. for u directly, not an averaging of U at neighbouring
    % ghost and interior cells
    xvel_FC         = setBoundaryConditions(xvel_FC,'xvel_FC');
    
    
    %_____________________________________________________
    % 5. Compute delta p
    % p satisfies a differential equation that can be
    % derived from the other ones plus the EOS. It looks
    % something like: p_t = -div(theta*u). Here we advance in time
    % p from tn to t_{n+1} using the advection operator of
    % theta*u.
    fprintf('Step 5: compute delta p\n');

    % Compute the advection operator of theta*u
    [ofs, rx] = outFluxVol(xvel_FC, delT, delX, nCells);            % Compute the outflux volumes ofs and centroid vector rx. Needs to be done once for all calls of advectQ in this timestep.
    [q_advected, gradLim, grad_x, rho_slab, rho_vrtx_1, rho_vrtx_2] = ...
        advectRho(volfrac_CC, ofs, rx, xvel_FC, delX, nCells);      % Treat theta advection like a rho advection (van Leer limiter, etc.)
    
    % Advance pressure in time
    delPDilatate(firstCell:lastCell) = (speedSound_CC(firstCell:lastCell).^2 ./ spvol_CC(firstCell:lastCell)) .* q_advected;
    
    press_CC(firstCell:lastCell) = press_CC(firstCell:lastCell) + delPDilatate(firstCell:lastCell);
    %   ^^----------------DOUBLE CHECK THIS SIGN!!!
    % Set boundary conditions on the new pressure
    press_CC =  setBoundaryConditions(press_CC,'press_CC');
           
    
    %_____________________________________________________
    % 6. Compute the face-centered pressure
    fprintf('Step 6: compute the face-centered pressure\n');
   
    for j = firstCell-1:lastCell                                    % Loop over all relevant cells
        this        = j;                                            % This face index
        adj         = j+1;                                          % The relevant adjacent face index
        press_FC(j) = (...
            press_CC(this) * rho_CC(adj ) + ...
            press_CC(adj ) * rho_CC(this)) / ...
            (rho_CC(this) + rho_CC(adj) + d_SMALL_NUM);
    end
    
    
    %_____________________________________________________    
    % 7. Accumulate sources
    % We neglect tortion forces, gravity effects, and the
    % temperature gradient term.
    fprintf('Step 7: accumulate sources\n');

    for j = firstCell:lastCell
        del_mom(j)  = -delT * delX * ...
            ((press_FC(j) - press_FC(j-1))/(delX));
            
        del_eng(j)  =  delX * press_CC(j) * delPDilatate(j) ...
                      /(speedSound_CC(j).^2 ./ spvol_CC(j));                % Shift q to the indices of the rest of the arrays in this formula
    end

    
    %_____________________________________________________
    % 8. Compute Lagrangian quantities
    fprintf('Step 8: compute Lagrangian quantities\n');

    mass_L      = rho_CC .* delX;                              % m^L = rho * cell_volume;
    mom_L       = mass_L .* xvel_CC + del_mom;              % (mu)^L = (mu) + del_(mu)
    eng_L       = mass_L .* cv .* temp_CC + del_eng;          % (me)^L = (me) + del_(me)

    % Translate fluxed quantities to primitive variables
    rho_CC      = mass_L ./ delX;
    xvel_CC     = mom_L ./ (mass_L + d_SMALL_NUM);
    temp_CC     = eng_L ./ (cv .* mass_L + d_SMALL_NUM);    
    
    % Set boundary conditions for primitive variables
    rho_CC      = setBoundaryConditions(rho_CC      ,'rho_CC');
    xvel_CC     = setBoundaryConditions(xvel_CC     ,'xvel_CC');
    temp_CC     = setBoundaryConditions(temp_CC     ,'temp_CC');
    
    % Translate primitives into fluxed quantities, now with the "B.C." set for them too
    mass_L      = rho_CC .* delX;
    mom_L       = mass_L .* xvel_CC;
    eng_L       = mass_L .* cv .* temp_CC;

    %_____________________________________________________
    % 9. Advect and advance in time
    fprintf('Step 9: advect and advance in time\n');
        
    %__________________________________
    % M A S S
    % Uses van Leer limiter
    fprintf ('Advecting density\n');
    [q_advected, gradLim, grad_x, rho_slab, rho_vrtx_1, rho_vrtx_2] = ...
        advectRho(mass_L, ofs, rx, xvel_FC, delX, nCells);      
    mass_CC(firstCell:lastCell)     = mass_L(firstCell:lastCell) - q_advected;                      % note: advection of rho * ofs (the advected volume) = advection correction to the mass
%    mass_CC(firstCell:lastCell)     = mass_L(firstCell:lastCell) - delT * q_advected;                      % note: advection of rho * ofs (the advected volume) = advection correction to the mass
    rho_CC      = mass_CC ./ delX;                                  % Updated density
    % Need to set B.C. on rho,T,u!!
    
    %__________________________________
    % M O M E N T U M
    % Uses van Leer limiter
    fprintf ('Advecting momentum\n');
    [q_advected, gradLim, grad_x] = ...
        advectQ(mom_L, rho_CC, rho_slab, rho_vrtx_1, rho_vrtx_2, ofs, rx, xvel_FC, delX, nCells);
    xvel_CC(firstCell:lastCell)     = (mom_L(firstCell:lastCell) - q_advected) ./ (mass_CC(firstCell:lastCell) + d_SMALL_NUM);  % Updated velocity

    %__________________________________
    % E N E R G Y
    % Uses van Leer limiter
    fprintf ('Advecting energy\n');
    [q_advected, gradLim, grad_x] = ...
        advectQ(eng_L, rho_CC, rho_slab, rho_vrtx_1, rho_vrtx_2, ofs, rx, xvel_FC, delX, nCells);
    temp_CC(firstCell:lastCell)     = (eng_L(firstCell:lastCell) - q_advected)./(cv.*mass_CC(firstCell:lastCell) + d_SMALL_NUM);% Updated temperature

    
    %_____________________________________________________
    % 10. End of the timestep

    %================ Compute del_T ================    
    % From ICE.cc: A G G R E S S I V E
    
    delt_CFL        = 1e+30;
    for j = firstCell:lastCell
        speed_Sound = speedSound_CC(j);
        A           = CFL*delX/(speed_Sound + abs(xvel_CC(j))+d_SMALL_NUM);
        delt_CFL    = min(A, delt_CFL);
    end
    fprintf('Aggressive delT Based on currant number CFL = %.3e\n',delt_CFL);

    %================ Plot results ================
    figure(1)
    subplot(2,2,1), plot(x_CC,rho_CC);
    xlim([boxLower(1) boxUpper(1)]);
    legend('\rho');
    grid on;

    subplot(2,2,2), plot(x_CC,xvel_CC);
    xlim([boxLower(1) boxUpper(1)]);
    legend('u_1');
    grid on;

    subplot(2,2,3), plot(x_CC,temp_CC);
    xlim([boxLower(1) boxUpper(1)]);
    legend('T');
    grid on;
    
    subplot(2,2,4), plot(x_CC,press_CC);
    xlim([boxLower(1) boxUpper(1)]);
    legend('p');
    grid on;
   
     M(tstep) = getframe(gcf);
    
    %================ Various breaks ================
    
    delT    = delt_CFL;                                         % Compute delT - "agressively" small
    t       = t + delT;                                         % Advance time
    if (t >= maxTime)
        fprintf('Reached maximum time\n');
        break;
    end
end

%hFig = figure(2);
%movie(hFig,M,1,10)
