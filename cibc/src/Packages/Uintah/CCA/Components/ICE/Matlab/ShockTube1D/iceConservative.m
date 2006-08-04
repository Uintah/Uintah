%ICE ICE Algorithm for the Shock Tube Problem in 1D (main driver) - 03/2005
%
%   This script tests full ICE time-stepping algorithm for Sod's Shocktube
%   problem in one space dimension.
%   We solve the compressible Euler equations for density (rho), veclocity
%   (u), temperature (T) and pressure (p). Internal energy is also used in
%   parts of the timestep (but is linearly related to the temperature).
%
%   Initial conditions:
%       rho=step function
%       u=0
%       T=300 Kelvin
%       p=constant
%
%   Reference:  "ICE, explicit pressure, single material, reaction model"
%               by Todd Harman, 09/24/04.
%               ICE input file: rieman_sm.ups
%
%   See also SETBOUNDARYCONDITIONS, ADVECTRHO, ADVECTQ.

%clear all;
close all;
globalParams;                                       % Load global parameters

%______________________________________________________________________
%     Problem Setup

%================ Set Parameters into struct P ================

% Geometry
P.boxLower          = 0;                            % Location of lower-left corner of domain
P.boxUpper          = 1;                            % Location of upper-right corner of domain

% Grid
P.nCells            = 100;                          % Number of cells in each direction
P.extraCells        = 1;                            % Number of ghost cells in each direction

% Time-stepping
P.maxTime           = 0.005;                        % Maximum simulation time [sec]
P.initTime          = 0.0;                          % Initial simulation time [sec]
P.delt_init         = 1e-6;                         % First timestep [sec]
P.maxTimeSteps      = 100;                          % Maximum number of timesteps [dimensionless]
P.CFL               = 0.45;                         % Courant number (~velocity*delT/delX) [dimensionless]
P.advectionOrder    = 1;                            % 1=1st-order advection operator; 2=possibly-limited-2nd-order

% Material properties (ideal gas)
P.cv                = 717.5;                        % Specific_heat
P.gamma             = 1.4;                          % gamma coefficient in the Equation of State (EOS)

%================ ICE Interal Parameters, Debugging Flags ================
% Debug flags
P.compareUintah     = 0;                            % Compares vs. Uintah ICE and plots results
P.debugSteps        = 0;                            % Debug printout of steps (tasks) within timestep
P.debugAdvectRho    = 0;                            % Debug printouts in advectRho()
P.debugAdvectQ      = 0;                            % Debug printouts in advectQ()
P.printRange        = [48:52];                      % Range of indices to be printed out (around the shock front at the first timestep, for testing)
P.plotInitialData   = 1;                            % Plots initial data

%================ Partition of the domain into regions ================

numRegions          = 2;                            % Partition of the domain into numRegions regions; physical properties are uniform within each region
Region              = cell(numRegions,1);           % This array holds the geometrical partition physical info
count               = 0;

%********** Parameters of the high density left region **********

count               = count+1;
R.label             = 'leftpartition';              % Title of this region
R.min               = 0;                            % Location of lower-left corner of this region [length]
R.max               = 0.5;                          % Location of upper-right corner of this region [length]
R.velocity          = 0.0;                          % Initial velocity
R.temperature       = 300.0;                        % Initial temperature [Kelvin]
R.density           = 1.1768292682926831000;        % Initial density
R.pressure          = 101325.0;                     % Initial pressure (1 atmosphere)
Region{count}       = R;                            % Add region to list

%********** Parameters of the low density right region **********

count               = count+1;
R.min               = 0.5;                          % Location of lower-left corner of this region [length]
R.max               = 1;                            % Location of upper-right corner of this region [length]
R.velocity          = 0.0;                          % Initial velocity
R.temperature       = 300.0;                        % Initial temperature [Kelvin]
R.density           = 0.11768292682926831000;       % Initial density
R.pressure          = 10132.50;                     % Initial pressure (0.1 atmosphere)
Region{count}       = R;                            % Add region to list

%______________________________________________________________________
%     Allocate arrays

%================ Cell Centered (CC) ================

totCells        = P.nCells + 2*P.extraCells;    % Array size for CC vars
x_CC            = zeros(1,totCells);            % Cell centers locations (x-component)
rho_CC          = zeros(1,totCells);            % Density rho
xvel_CC         = zeros(1,totCells);            % Velocity u (x-component)
temp_CC         = zeros(1,totCells);            % Temperature T
press_eq_CC     = zeros(1,totCells);            % equilibraton pressure
press_CC        = zeros(1,totCells);            % Pressure p
volfrac_CC      = zeros(1,totCells);            % Volume fraction theta, = 1 for this single-material problem
spvol_CC        = zeros(1,totCells);            % Specific volume v0 = theta/rho = 1/(specific density)
mass_CC         = zeros(1,totCells);            % Mass at time n+1 (see step 9)

gradLim         = zeros(1,totCells);            % Gradient Limiter (for advection)
grad_x          = zeros(1,totCells);            % Gradient of rho (for advection)
q_advected      = zeros(1,totCells);            % Advected flux of quantity (for advection)
mass_slab       = zeros(1,totCells);            % Mass in slab moved in/out a cell in advection
delPDilatate    = zeros(1,totCells);            % Pressure increment due to dilatation

%================ Cell Centered (CC), Lagrangian values (L) ================

del_mom         = zeros(1,totCells);            % Momentum accumulated sources
del_eng         = zeros(1,totCells);            % Energy accumulated sources
mass_L          = zeros(1,totCells);            % Mass ( = rho * cell volume )
mom_L           = zeros(1,totCells);            % Momentum ( = mass * velocity )
eng_L           = zeros(1,totCells);            % Energy (= mass * internal energy = mass * cv * temperature)

%================ Face Centered (FC) ================

totFaces        = P.nCells + 1;                 % Array size for FC vars
xvel_FC         = zeros(1,totFaces);            % Velocity u (x-component)
press_FC        = zeros(1,totFaces);            % Pressure p

%================ Node Centered (NC) ================

totNodes        = P.nCells + 1;                 % Array size for NC vars
mass_vrtx_1     = zeros(1,totNodes);            % Mass at vertices (for advection)
mass_vrtx_2     = zeros(1,totNodes);            % --------//-------  (for advection)

%______________________________________________________________________
%     Initialization

%================ Useful constants ================

d_SMALL_NUM = 1e-100;                           % A small number (for bullet-proofing)
nCells      = P.nCells;                         % # interior cells
delX        = (P.boxUpper-P.boxLower)./nCells;  % Cell length
delT        = P.delt_init;                      % Init timestep
ghost_Left  = 1;                                % Index of left ghost cell
ghost_Right = P.nCells+2*P.extraCells;          % Index of right ghost cell
firstCell   = 2;                                % Index of first interior cell
lastCell    = nCells+1;                         % Index of last interior cell
if (P.compareUintah)                            % Don't make more than 2 timsteps when comparing to Uintah
    P.maxTimeSteps = min(P.maxTimeSteps,2);
end

%================ Initialize interior cells ================
% Initial data at t=0 in the interior domain.

x_CC    = ([1:totCells]-1-0.5).*delX;           % Cell centers coordinates (note the "1-based" matlab index array)
for r = 1:numRegions                            % Init each region, assuming they are a non-overlapping all-covering partition of the domain
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
if (P.plotInitialData)
    figure(1);
    set(gcf,'position',[100,600,400,400]);
    %================ Plot results ================

    subplot(2,2,1), plot(x_CC,rho_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('\rho');
    grid on;

    subplot(2,2,2), plot(x_CC,xvel_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('u_1');
    grid on;

    subplot(2,2,3), plot(x_CC,temp_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('T');
    grid on;

    subplot(2,2,4), plot(x_CC,press_CC);
    xlim([P.boxLower(1) P.boxUpper(1)]);
    legend('p');
    grid on;

    %M(tstep) = getframe(gcf);
    print -depsc iceInitialData.eps
end

%______________________________________________________________________
%     Time integration loop
t = P.initTime + delT;
for tstep = 1:P.maxTimeSteps
    fprintf('\n_____________________________________tstep=%d, t=%e, prev. delT=%e\n', tstep, t, delT);

    %_____________________________________________________
    % 0. Dummy setting for a single-material problem
    % Set the volume fraction and specific volume.

    if (P.debugSteps)
        fprintf('Step 0: dummy setting for single-material\n');
    end

    volfrac_CC      = ones(1,totCells);                             % Single material ==> covers 100% of each cell (volfrac=1)
    spvol_CC        = volfrac_CC ./ (rho_CC + d_SMALL_NUM);         % Specific volume, here also = 1/rho


    %_____________________________________________________
    % 1. Compute thremodynamic/transport properties
    % These are constants in this application and were already computed in the
    % initialization stage.


    %_____________________________________________________
    % 2. Compute the equilibration pressure
    % Using an equation of State for ideal gas.
    if (P.debugSteps)
        fprintf('Step 2: compute equilibration pressure\n');
    end

    % Update pressure from EOS
    press_eq_CC      = (P.gamma-1).*P.cv.*rho_CC.*temp_CC;          % Evaluate p from EOS

    % Compute speed of sound at cell centers
    DpDrho          = (P.gamma-1).*P.cv.*temp_CC;                   % d P / d rho
    DpDe            = (P.gamma-1).*rho_CC;                          % d P / d e
    tmp             = DpDrho + ( DpDe .* (press_eq_CC./rho_CC.^2));
    speedSound_CC   = sqrt(tmp);                                    % Speed of sound


    % Set boundary conditions on p
    press_eq_CC     =  setBoundaryConditions(press_eq_CC,'press_CC');


    %_____________________________________________________
    % 3. Compute sources of energy due to chemical reactions
    % Not applicable to this model.


    %_____________________________________________________
    % 4. Compute the face-centered velocities
    if (P.debugSteps)
        fprintf('Step 4: compute face-centered velocities\n');
    end

    for j = firstCell-1:lastCell                                    % Loop over all faces
        left        = j;
        right       = j+1;
        term1       = (...
            rho_CC(left )*xvel_CC(left ) + ...
            rho_CC(right)*xvel_CC(right) ...
            ) / (rho_CC(left) + rho_CC(right) + d_SMALL_NUM);

        term2a      = 2.0*spvol_CC(left)*spvol_CC(right) ...
            / (spvol_CC(left) + spvol_CC(right) + d_SMALL_NUM);

        term2b      = (press_eq_CC(right) - press_eq_CC(left))/delX;

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
    if (P.debugSteps)
        fprintf('Step 5: compute delta p\n');
    end

    [ofs, rx] = outFluxVol(xvel_FC, delT, delX, nCells);            % Compute the outflux volumes ofs and centroid vector rx. Needs to be done once for all calls of advectQ in this timestep.
    [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ...     % Compute q_advected = -Delta t ADV(vol_frac,u^*)
        advectRho(volfrac_CC, ofs, rx, xvel_FC, delX, nCells);      % Treat vol frac (theta) advection like a mass advection (van Leer limiter, etc.)
    term1 = (speedSound_CC.^2) ./ spvol_CC;                         % Measure of flow compressibility
    delPDilatate = term1 .* q_advected;
    press_CC = press_eq_CC + delPDilatate;                          % Advance pressure in time
    press_CC =  setBoundaryConditions(press_CC,'press_CC');         % Set boundary conditions on the new pressure


    %_____________________________________________________
    % 6. Compute the face-centered pressure
    if (P.debugSteps)
        fprintf('Step 6: compute the face-centered pressure\n');
    end

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
    if (P.debugSteps)
        fprintf('Step 7: accumulate sources\n');
    end

    for j = firstCell:lastCell
        del_mom(j)  = -delT * delX * ...
            ((press_FC(j) - press_FC(j-1))/delX);

%         del_eng(j)  =  delX .*spvol_CC(j) .* press_CC(j) * delPDilatate(j) ...
%             /(speedSound_CC(j).^2);                                 % Shift q to the indices of the rest of the arrays in this formula
    end

    % Total-energy Lagrangian delta-term
    [ofs, rx] = outFluxVol(xvel_FC, delT, delX, nCells);            % Compute the outflux volumes ofs and centroid vector rx. Needs to be done once for all calls of advectQ in this timestep.
    [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ... 
        advectRho(press_CC, ofs, rx, xvel_FC, delX, nCells);        % Compute q_advected = -Delta t ADV(p^{n+1},u^*) . Limiter??
    del_eng = delX.*q_advected;
    
    %_____________________________________________________
    % 8. Compute Lagrangian quantities
    if (P.debugSteps)
        fprintf('Step 8: compute Lagrangian quantities\n');
    end

    mass_L      = rho_CC .* delX;                                   % m^L = rho * cell_volume;
    mom_L       = mass_L .* xvel_CC + del_mom;                      % (mu)^L = (mu) + del_(mu)
    eng_L       = mass_L .* P.cv .* temp_CC + mom_L .* xvel_CC + del_eng;              % (me)^L = mi + m u^2 + del_(me)

    % Translate fluxed quantities to primitive variables
    rho_CC      = mass_L ./ delX;
    xvel_CC     = mom_L ./ (mass_L + d_SMALL_NUM);
    temp_CC     = (eng_L - mom_L .* xvel_CC)./ (P.cv .* mass_L + d_SMALL_NUM);

    % Set boundary conditions for primitive variables
    rho_CC      = setBoundaryConditions(rho_CC      ,'rho_CC');
    xvel_CC     = setBoundaryConditions(xvel_CC     ,'xvel_CC');
    temp_CC     = setBoundaryConditions(temp_CC     ,'temp_CC');

    % Translate primitives into fluxed quantities, now with the "B.C." set for them too
    mass_L      = rho_CC .* delX;
    mom_L       = mass_L .* xvel_CC;
    eng_L       = mass_L .* P.cv .* temp_CC + mom_L .* xvel_CC;

    %_____________________________________________________
    % 9. Advect and advance in time
    if (P.debugSteps)
        fprintf('Step 9: advect and advance in time\n');
    end

    %==================================
    % M A S S
    % Uses van Leer limiter
    if (P.debugSteps)
        fprintf ('Advecting density\n');
    end
    [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ...
        advectRho(mass_L, ofs, rx, xvel_FC, delX, nCells);
    mass_CC     = mass_L + q_advected;                              % Advection of rho*ofs (the advected volume) = advection correction to the mass
    rho_CC      = mass_CC ./ delX;                                  % Updated density
    rho_CC      = setBoundaryConditions(rho_CC      ,'rho_CC');     % We need to set B.C. on rho,T,u
    
    %==================================
    % M O M E N T U M
    % Uses compatible flux limiter
    if (P.debugSteps)
        fprintf ('Advecting momentum\n');
    end
    [q_advected, gradLim, grad_x] = ...
        advectQ(mom_L, mass_L, mass_slab, mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, delX, nCells);



    xvel_CC     = (mom_L + q_advected) ./ (mass_CC);                % Updated velocity
    xvel_CC     = setBoundaryConditions(xvel_CC     ,'xvel_CC');

    %==================================
    % E N E R G Y
    % Uses compatible flux limiter
    if (P.debugSteps)
        fprintf ('Advecting energy\n');
    end
    [q_advected, gradLim, grad_x] = ...
        advectQ(eng_L, mass_L, mass_slab, mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, delX, nCells);
    temp_CC     = (eng_L - mom_L .* xvel_CC + q_advected)./(P.cv.*mass_CC);            % Updated temperature
    temp_CC     = setBoundaryConditions(temp_CC     ,'temp_CC');


    %_____________________________________________________
    % 10. End of the timestep

    %================ Compute del_T ================
    % A G G R E S S I V E

    delt_CFL        = 1e+30;
    for j = firstCell:lastCell
        speed_Sound = speedSound_CC(j);
        A           = P.CFL*delX/(speed_Sound + abs(xvel_CC(j)));
        delt_CFL    = min(A, delt_CFL);
    end
    if (P.debugSteps)
        fprintf('Aggressive delT Based on currant number CFL = %.3e\n',delt_CFL);
    end


    %================ Compare with Uintah ICE and Plot Results ================
    if (P.compareUintah)
        loadUintah;                                                 % Load Uintah results
    end
    plotResults;                                                    % Plot results (and also against Uintah, if compareUintah flag is on)

    %================ Various breaks ================

    delT    = delt_CFL;                                             % Compute delT - "agressively" small
    t       = t + delT;                                             % Advance time
    if (t >= P.maxTime)
        fprintf('Reached maximum time\n');
        break;
    end
end
figure(1);
print -depsc iceResult1.eps
figure(2);
print -depsc iceResult2.eps

% Show a movie of the results
%hFig = figure(2);
%movie(hFig,M,1,10)
