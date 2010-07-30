%ICE ICE Algorithm for the Shock Tube Problem in 1D (main driver) - 03/2005
%
%   This script tests full ICE time-stepping algorithm for Sod's Shocktube
%   problem in one space dimension.
%   We solve the compressible Euler equations for density (rho), velocity
%   (u), internal energy (e) and pressure (p). 
%
%   Reference:  "ICE, explicit pressure, single material, reaction model"
%               by Todd Harman, 09/24/04.
%               ICE input file: rieman_sm.ups
%
%   See also SETBOUNDARYCONDITIONS, ADVECTRHO, ADVECTQ.


function [tfinal, x_CC, delX, rho_CC, xvel_CC, press_CC, temp_CC]=ice(nCells)


close all;
globalParams;                                       % Load global parameters
setenv('LD_LIBRARY_PATH', ['/usr/lib']);
[OF] = outputFunctions;
%______________________________________________________________________
%     Problem Setup

%================ Set Parameters into struct P ================

% Geometry
P.boxLower          = 0;                            % Location of lower-left corner of domain
P.boxUpper          = 1;                            % Location of upper-right corner of domain

% Grid
P.nCells            = nCells;                       % Number of cells in each direction
P.extraCells        = 1;                            % Number of ghost cells in each direction

% Time-stepping
P.initTime          = 0.0;                          % Initial simulation time [sec]
P.writeData         = 1;                            % output the final timestep to a .dat file
P.delt_init         = 1e-20;                        % First timestep [sec]
P.maxTimeSteps      = 100000                           % Maximum number of timesteps [dimensionless]
P.UseProbeCells     = 1;                            % on/off switch for writing out data at probe locations
P.probeLocations    = [ 0.4 0.5 0.6]
                                                    % physical locations of probe locations

P.CFL               = 0.25;                         % Courant number (~velocity*delT/delX) [dimensionless]
P.advectionOrder    = 2;                            % 1=1st-order advection operator; 2=possibly-limited-2nd-order

[P, Region, numRegions] = initialConditions()

%================ ICE Interal Parameters, Debugging Flags ================
% Debug flags
P.compareUintah     = 0;                            % Compares vs. Uintah ICE and plots results
P.debugSteps        = 0;                            % Debug printout of steps (tasks) within timestep
P.debugAdvectRho    = 0;                            % Debug printouts in advectRho()
P.debugAdvectQ      = 0;                            % Debug printouts in advectQ()
P.printRange        = [27:32];                      % Range of indices to be printed out (around the shock front at the first timestep, for testing)
P.plotInitialData   = 0;                            % Plots initial data
P.plotResults       = 1;                            % plot intermediate data
P.plotInterval      = 100;                          % plot every n timesteps

%================ Grid Struct (G) ======= ================
G.nCells      = P.nCells;                           % # interior cells
G.delX        = (P.boxUpper-P.boxLower)./G.nCells;  % Cell length
G.ghost_Left  = 1;                                  % Index of left ghost cell
G.ghost_Right = G.nCells+2*P.extraCells;            % Index of right ghost cell
G.first_CC    = 2;                                  % Index of first interior cell
G.first_FC    = 2;                                  % Index of first interior xminus face
G.last_CC     = G.nCells+1;                         % Index of last interior cell
G.last_FC     = G.ghost_Right;                      % index of last xminus face


%______________________________________________________________________
%     Allocate arrays

%================ Cell Centered (CC) ================

totCells        = P.nCells + 2*P.extraCells;              % Array size for CC vars
x_CC            = zeros(G.ghost_Left,G.ghost_Right);      % Cell centers locations (x-component)
rho_CC          = zeros(G.ghost_Left,G.ghost_Right);      % Density rho
xvel_CC         = zeros(G.ghost_Left,G.ghost_Right);      % Velocity u (x-component)
temp_CC         = zeros(G.ghost_Left,G.ghost_Right);      % Temperature T
press_eq_CC     = zeros(G.ghost_Left,G.ghost_Right);      % equilibraton pressure
press_CC        = zeros(G.ghost_Left,G.ghost_Right);      % Pressure p
volfrac_CC      = zeros(G.ghost_Left,G.ghost_Right);      % Volume fraction theta, = 1 for this single-material problem
spvol_CC        = zeros(G.ghost_Left,G.ghost_Right);      % Specific volume v0 = theta/rho = 1/(specific density)
mass_CC         = zeros(G.ghost_Left,G.ghost_Right);      % Mass at time n+1 (see step 9)
delPDilatate    = zeros(G.ghost_Left,G.ghost_Right);      % Pressure increment due to dilatation

%================ Cell Centered (CC), Lagrangian values (L) ================

del_mom         = zeros(G.ghost_Left,G.ghost_Right);      % Momentum accumulated sources
del_eng         = zeros(G.ghost_Left,G.ghost_Right);      % Energy accumulated sources
mass_L          = zeros(G.ghost_Left,G.ghost_Right);      % Mass ( = rho * cell volume )
mom_L           = zeros(G.ghost_Left,G.ghost_Right);      % Momentum ( = mass * velocity )
eng_L           = zeros(G.ghost_Left,G.ghost_Right);      % Energy (= mass * internal energy = mass * cv * temperature)

%================ Face Centered (FC) ================
x_FC            = zeros(G.ghost_Left,G.last_FC);          % Face centers locations (x-component)  
xvel_FC         = zeros(G.ghost_Left,G.last_FC);          % Velocity u (x-component)              
press_FC        = zeros(G.ghost_Left,G.last_FC);          % Pressure p                            
speedSound_FC   = zeros(G.ghost_Left,G.last_FC);          % c^*                                   

%================ Node Centered (NC) ================
totNodes        = P.nCells + 1;                           % Array size for NC vars
mass_vrtx_1     = zeros(1,totNodes);                      % Mass at vertices (for advection)
mass_vrtx_2     = zeros(1,totNodes);                      % --------//-------  (for advection)

%______________________________________________________________________
%     Initialization
d_SMALL_NUM = 1e-100;                           % A small number (for bullet-proofing
d_TINY_RHO  = 1.0e-12;
delT        = P.delt_init;                      % Init timestep

%================ Initialize interior cells ================
% Initial data at t=0 in the interior domain.
x_CC    = ([G.ghost_Left:G.ghost_Right]-1-0.5).*G.delX + P.boxLower; 
x_FC(1) = -G.delX + P.boxLower;

for j = G.first_FC:G.last_FC   % Loop over all xminus cell faces
  x_FC(j) = (j - G.first_FC).*G.delX + P.boxLower;  
end

for r = 1:numRegions                                    % Init each region, assuming they are a non-overlapping all-covering partition of the domain
  R       = Region{r};
  first   = max(G.first_CC,floor(R.min./G.delX  + G.first_CC));
  last    = min(G.last_CC, floor(R.max./G.delX  + G.first_CC));

  for j = first:last
     rho_CC(j)   = R.density + d_TINY_RHO * R.density;
    %rho_CC(j)    = 1.2 + 0.5*exp(-((x_CC(j)- 0.5)^2)/.01);
    xvel_CC(j)   = R.velocity;
    temp_CC(j)   = R.temperature;
    press_CC(j)  = (P.gamma -1)*P.cv*R.temperature*rho_CC(j);
  end
end

%================ Boundary Conditions ================
rho_CC      = setBoundaryConditions(rho_CC  ,'rho_CC',   G);
xvel_CC     = setBoundaryConditions(xvel_CC ,'xvel_CC',  G);
temp_CC     = setBoundaryConditions(temp_CC ,'temp_CC',  G);
press_CC    = setBoundaryConditions(press_CC,'press_CC', G);

%================ Plot Initial conditions ================
if (P.plotInitialData)
  fig1 = sfigure(1);
  set(fig1,'position',[100,1000,1000,400]);

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
  pause
end

% find the cell indicies of the probe cells
if(P.UseProbeCells)
  [P.probeCells] = findProbeCellIndices(P, x_CC)
end

%______________________________________________________________________
%     Time integration loop
t = P.initTime + delT;
tstep = 0;
for tstep = 1:P.maxTimeSteps
%while (tstep <= P.maxTimeSteps & t <= P.maxTime)
  %tstep = tstep + 1;
  fprintf('\n_____________________________________tstep=%d, t=%e, prev. delT=%e\n', tstep, t, delT);

  %_____________________________________________________
  % Set the volume fraction and specific volume.
  if (P.debugSteps)
    fprintf('Step 0: compute volFrac\n');
  end

  volfrac_CC = ones(G.ghost_Left,G.ghost_Right);             % Single material ==> covers 100% of each cell (volfrac=1)
  spvol_CC   = volfrac_CC ./ (rho_CC + d_SMALL_NUM);         % Specific volume, here also = 1/rho


  %_____________________________________________________
  % 1. Compute thremodynamic/transport properties
  % These are constants for now

  %_____________________________________________________
  % 2. Compute the equilibration pressure
  % Using an equation of State for ideal gas.
  if (P.debugSteps)
    fprintf('Step 2: compute equilibration pressure\n');
  end

  % Update pressure from EOS
  press_eq_CC      = (P.gamma-1.0).*P.cv.*rho_CC.*temp_CC;          % Evaluate p from EOS

  % Compute speed of sound at cell centers
  DpDrho          = (P.gamma-1.0).*P.cv.*temp_CC;                   % d P / d rho
  DpDe            = (P.gamma-1.0).*rho_CC;                          % d P / d e
  tmp             = DpDrho + ( DpDe .* (press_eq_CC./rho_CC.^2));
  speedSound_CC   = sqrt(tmp);                                      % Speed of sound


  press_eq_CC     =  setBoundaryConditions(press_eq_CC,'press_CC',G);

  %_____________________________________________________
  % 3. Compute sources of energy due to chemical reactions
  % Not applicable to this model.


  %_____________________________________________________
  % 4. Compute the face-centered velocities
  if (P.debugSteps)
    fprintf('Step 4: compute face-centered velocities\n');
  end

  for j = G.first_FC:G.last_FC   % Loop over all xminus cell faces
    L = j-1;
    R = j;
    term1   = (rho_CC(L)*xvel_CC(L) + rho_CC(R)*xvel_CC(R)) / (rho_CC(L) + rho_CC(R));

    term2a  = (2.0*spvol_CC(L)*spvol_CC(R)) / (spvol_CC(L) + spvol_CC(R));

    term2b  = (press_eq_CC(R) - press_eq_CC(L))/G.delX;
    
    xvel_FC(j)  = term1 - delT*(term2a*term2b);
  end

  % Set boundary conditions on u @ face centers
  xvel_FC = setBoundaryConditions(xvel_FC,'xvel_FC', G);


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

  % Compute the advection operator of theta*u
  [ofs, rx] = outFluxVol(xvel_CC, xvel_FC, delT, G);                  % Compute the outflux volumes ofs and centroid vector rx. Needs to be done once for all calls of advectQ in this timestep.
  
  [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ...
    advectRho(volfrac_CC, ofs, rx, xvel_FC, G);                       % Treat vol frac (theta) advection like a mass advection (van Leer limiter, etc.)

  for j = G.first_CC:G.last_CC
    term1 = ((speedSound_CC(j)^2) / spvol_CC(j) );

    delPDilatate(j) = term1 * q_advected(j);
    press_CC(j) = press_eq_CC(j) + delPDilatate(j);                   % Advance pressure in time
  end
  press_CC =  setBoundaryConditions(press_CC,'press_CC',G);           % Set boundary conditions on the new pressure


  %_____________________________________________________
  % 6. Compute the face-centered pressure
  if (P.debugSteps)
    fprintf('Step 6: compute the face-centered pressure\n');
  end

  for j = G.first_FC:G.last_FC                                   
    L  = j-1; 
    R = j;
    press_FC(j) = (press_CC(L)* rho_CC(R) + press_CC(R) * rho_CC(L)) / ...
                   (rho_CC(L) + rho_CC(R) + d_SMALL_NUM);
  end


  %_____________________________________________________
  % 7. Accumulate sources
  % We neglect tortion forces, gravity effects, and the
  % temperature gradient term.
  if (P.debugSteps)
    fprintf('Step 7: accumulate sources\n');
  end
  
  for j = G.first_CC:G.last_CC
    L = j;
    R = j+1;
  
    del_mom(j)  = -delT * G.delX * ...
      ((press_FC(R) - press_FC(L))/G.delX);

    del_eng(j)  =  G.delX * spvol_CC(j) * press_CC(j) * delPDilatate(j) ...
      /(speedSound_CC(j)^2);
  end


  %_____________________________________________________
  % 8. Compute Lagrangian quantities
  if (P.debugSteps)
    fprintf('Step 8: compute Lagrangian quantities\n');
  end

  mass_L      = rho_CC .* G.delX;                                 % m^L = rho * cell_volume;
  mom_L       = mass_L .* xvel_CC + del_mom;                      % (mu)^L = (mu) + del_(mu)
  eng_L       = mass_L .* P.cv .* temp_CC + del_eng;              % (me)^L = (me) + del_(me)

  % Translate fluxed quantities to primitive variables
  rho_CC      = mass_L ./ G.delX;
  xvel_CC     = mom_L ./ (mass_L + d_SMALL_NUM);
  temp_CC     = eng_L ./ (P.cv .* mass_L + d_SMALL_NUM);

  % Set boundary conditions for primitive variables
  rho_CC      = setBoundaryConditions(rho_CC, 'rho_CC',  G);
  xvel_CC     = setBoundaryConditions(xvel_CC,'xvel_CC', G);
  temp_CC     = setBoundaryConditions(temp_CC,'temp_CC', G);

  % Translate primitives into fluxed quantities, now with the "B.C." set for them too
  mass_L      = rho_CC .* G.delX;
  mom_L       = mass_L .* xvel_CC;
  eng_L       = mass_L .* P.cv .* temp_CC;

  %_____________________________________________________
  % 9. Advect and advance in time
  if (P.debugSteps)
    fprintf('Step 9: advect and advance in time\n');
  end

  % Mass:  Uses van Leer limiter
  [q_advected, gradLim, grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ...
    advectRho(mass_L, ofs, rx, xvel_FC, G);
  
  mass_CC     = mass_L + q_advected;                        % Advection of rho*ofs (the advected volume) = advection correction to the mass
  rho_CC      = mass_CC ./ G.delX;                          % Updated density
  rho_CC      = setBoundaryConditions(rho_CC,'rho_CC', G); 

  % Momentum: Uses compatible flux limiter
  [q_advected, gradLim, grad_x] = ...
    advectQ(mom_L, mass_L, mass_slab, mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, G);
    
  xvel_CC     = (mom_L + q_advected) ./ (mass_CC);           % Updated velocity
  xvel_CC     = setBoundaryConditions(xvel_CC,'xvel_CC', G);

  % Energy:  Uses compatible flux limiter
  [q_advected, gradLim, grad_x] = ...
    advectQ(eng_L, mass_L, mass_slab, mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, G);
  
  temp_CC     = (eng_L + q_advected)./(P.cv.*mass_CC);            % Updated temperature
  temp_CC     = setBoundaryConditions(temp_CC,'temp_CC', G);


  %_____________________________________________________
  % End of the timestep


  CC_data = createDataStruct(x_CC, press_CC, press_eq_CC, delPDilatate, xvel_CC, temp_CC, rho_CC);
  FC_data = createDataStruct(x_FC, xvel_FC, press_FC );
  
  
  if(P.UseProbeCells)
    for c = 1:length(P.probeCells)
      cell = P.probeCells(c);
      filename = sprintf('probeCell_%g.dat', P.probeLocations(c));
      OF.writeProbePoints(filename,tstep, t, CC_data, cell);
    end
  end
   
  %  Compute delT
  if (t >= P.maxTime)   
    fprintf('Reached maximum time\n');
    break;
  end

  delT        = 1e+30;
  for j = G.first_CC:G.last_CC
    speed_Sound = speedSound_CC(j);
    A           = P.CFL*G.delX/(speed_Sound + abs(xvel_CC(j)));
    delT    = min(A, delT);
  end
  
  if (P.debugSteps)
    fprintf('Aggressive delT Based on currant number CFL = %.3e\n',delT);
  end

  %================ Compare with Uintah ICE and Plot Results ================
  if (P.compareUintah)
    loadUintah;                             % Load Uintah results
  end
  plotResults;                              % Plot results (and also against Uintah, if compareUintah flag is on)

 % delT   = 5e-4;
 % fprintf ('-------------WARNING:   delT has been hard coded to compare with Uintah results\n');
  
  t = t + delT;                             % Advance time
  if (t >= P.maxTime)
    fprintf('Reached maximum time\n');
    break;
  end

end

  tfinal = t - delT;
  delX   = G.delX;
  
  
  % __________________________________
  %   OUTPUT
  if (P.writeData == 1)
    CCfilename = sprintf('matlab_CC_%g.dat', P.nCells);
    FCfilename = sprintf('matlab_FC_%g.dat', P.nCells);
    OF.writeData( CCfilename, t-delT, CC_data);
    OF.writeData( FCfilename, t-delT, FC_data);
  end

  if(P.plotResults)
    plotResults;
    sfigure(1);
    print -depsc iceResult1.eps
    sfigure(2);
    print -depsc iceResult2.eps
  end
  % Show a movie of the results
  %hFig = sfigure(2);
  %movie(hFig,M,1,10)

end  % end of ICE


%__________________________________
function [data] =  createDataStruct(varargin)

  for k = 1:length(varargin)    % loop over each input argument
    varG = varargin{k};
    name = inputname(k);
    
    data.(name)=varG;
  end
end

%__________________________________
% This function computes probeCell indices
function [cell] = findProbeCellIndices(P, x_CC)

  for p = 1:length(P.probeLocations)
    x = P.probeLocations(p);
    diff = abs(x - x_CC);
    [tmp,cell(p)] = min(diff);
  end
end
