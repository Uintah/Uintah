% Reference:  Jin Ma, Honbing Lu,and Ranga Komanduri, "Structured Mesh
% Refinement in Generalized Interpolation Material Point (GIMP) Method
% for Simulation of Dynamic Problems," CMES, vol. 12, no.3, pp. 213-227, 2006
%______________________________________________________________________
                                                         
   
   

function [L2_norm,maxError,NN,NP]=ml_amrmpm(problem_type,CFL,NCells)

close all
intwarning on
addpath([docroot '/techdoc/creating_plots/examples'])   % so you can use dsxy2figxy()


global d_debugging;
global PPC;
global sf;
global NSFN;               % number of shape function nodes
d_debugging = problem_type;

[mms] = MMS;                      % load mms functions
[sf]  = shapeFunctions;           % load all the shape functions
[IF]  = initializationFunctions   % load initialization functions

%  valid options:
%  problem type:  impulsiveBar, oscillator, compaction advectBlock, mms
%  CFL:            0 < cfl < 1, usually 0.2 or so.

% bulletproofing
if (~strcmp(problem_type, 'impulsiveBar')  && ...
    ~strcmp(problem_type, 'oscillator')    && ...
    ~strcmp(problem_type, 'compaction')    && ...
    ~strcmp(problem_type, 'advectBlock')   && ...
    ~strcmp(problem_type, 'mms' ))
  fprintf('ERROR, the problem type is invalid\n');
  fprintf('     The valid types are impulsiveBar, oscillator, advectBlock, compaction\n');
  fprintf('usage:  amrmpm(problem type, cfl, L1_dx)\n');
  return;
end
%__________________________________
% Global variables

PPC     = 2;
E       = 1.0e4;
density = 1.0;
speedSound = sqrt(E/density);

Material = cell(1,1);     % structure that holds material properties
mat.E           = E;
mat.density     = density;
mat.speedSound  = speedSound;
Material{1} = mat;
Material{1}.density

interpolation = 'LINEAR';

if( strcmp(interpolation,'GIMP') )
  NSFN    = 3;               % Number of shape function nodes Linear:2, GIMP:3
else
  NSFN    = 2;        
end

t_initial  = 0.0;             % physical time
tstep     = 0;                % timestep
bodyForce = 0;

BigNum     = int32(1e5);
d_smallNum = double(1e-16);

bar_min     = 0.0;
bar_max     = 1;
bar_length  = bar_max - bar_min;

domain       = 1.0;
area         = 1.;
plotSwitch   = 1;
plotInterval = 100;
writeData    = 0;
max_tstep    = BigNum;

L1_dx       =domain/(NCells)

if (mod(domain,L1_dx) ~= 0)
  fprintf('ERROR, the dx in Region 1 does not divide into the domain evenly\n');
  fprintf('usage:  amrmpm(problem type, cfl, L1_dx)\n');
  return;
end

%__________________________________
% Grid Initialization
% compute the zone of influence
% compute the positions of the nodes
[Levels, dx_min, Limits] = IF.initialize_grid(domain,PPC,L1_dx,interpolation, d_smallNum);

[nodePos]                = IF.initialize_NodePos(L1_dx, Levels, Limits, interpolation);

[Lx]                     = IF.initialize_Lx(nodePos, Levels, Limits);

[xp, Levels, Limits]     = IF.initialize_xp(nodePos, interpolation, PPC, bar_min, bar_max, Limits, Levels);

[Limits]                 = IF.NN_NP_allLevels(Levels, Limits)


% define the boundary condition nodes on Level 1
L1_NN = Levels{1}.NN
BCNodeL(1)  = 1;
BCNodeR(1)  = L1_NN;

if(strcmp(interpolation,'GIMP'))
  BCNodeL(1) = 1;
  BCNodeL(2) = 2;
  BCNodeR(1) = L1_NN-1;
  BCNodeR(2) = L1_NN;
end
%__________________________________
% pre-allocate variables for speed
maxLevels = Limits.maxLevels;
NP_max    = Limits.NP_max;
NN_max    = Limits.NN_max;

vol       = zeros(NP_max, maxLevels);
lp        = zeros(NP_max, maxLevels);
massP     = zeros(NP_max, maxLevels);
velP      = zeros(NP_max, maxLevels);
dp        = zeros(NP_max, maxLevels);
stressP   = zeros(NP_max, maxLevels);
Fp        = zeros(NP_max, maxLevels);
dF        = zeros(NP_max, maxLevels);
accl_extForceP = zeros(NP_max,maxLevels);

nodes     = zeros(1,NSFN);
Gs        = zeros(1,NSFN);
Ss        = zeros(1,NSFN);

massG     = zeros(NN_max,  maxLevels);
momG      = zeros(NN_max,  maxLevels);
velG      = zeros(NN_max,  maxLevels);
vel_nobc_G= zeros(NN_max,  maxLevels);
accl_G    = zeros(NN_max,  maxLevels);
extForceG = zeros(NN_max,  maxLevels);
intForceG = zeros(NN_max,  maxLevels);


KE        = zeros(BigNum, maxLevels);
SE        = zeros(BigNum, maxLevels);
TE        = zeros(BigNum, maxLevels);
totalMom  = zeros(BigNum, maxLevels);
Exact_tip = zeros(BigNum, maxLevels);
DX_tip    = zeros(BigNum, maxLevels);
TIME      = zeros(BigNum, maxLevels);

totalEng_err   = zeros(BigNum, maxLevels);
totalMom_err   = zeros(BigNum, maxLevels);
tipDeflect_err = zeros(BigNum, maxLevels);

input('hit return')

%__________________________________
% initialize other particle variables
for l=1:maxLevels
   L = Levels{l};
   
  for ip=1: L.NP
    [volP_0, lp_0] = sf.positionToVolP(xp(ip,l), L.nPatches, L.Patches);

    vol(ip,l)   = volP_0;
    massP(ip,l) = volP_0*density;
    lp(ip,l)    = lp_0;
    Fp(ip,l)    = 1.;                     % total deformation
  end
end 

%______________________________________________________________________
% problem specific parameters

if strcmp(problem_type, 'impulsiveBar')
  period          = sqrt(16.*bar_length*bar_length*density/E);
  t_final          = period;
  TipForce        = 10.;
  D               = TipForce*bar_length/(area*E);
  M               = 4.*D/period;
  accl_extForceP(NP)   = TipForce;
  velG_BCValue(1) = 0.;
  velG_BCValue(2) = 1.;
end

if strcmp(problem_type, 'oscillator')
  Mass            = 10000.;
  period          = 2.*3.14159/sqrt(E/Mass);
  t_final          = period;
  v0              = 0.5;
  Amp             = v0/(2.*3.14159/period);
  massP(NP)       = Mass;                    % last particle masss
  velP(NP)        = v0;                      % last particle velocity
  numBCs          = 1;
  velG_BCValueL   = 0.;
  velG_BCValueR   = 1.;
end

if strcmp(problem_type, 'advectBlock')
  initVelocity    = 100;
  t_final          = 0.5;
  numBCs          = 1;
  velG_BCValueL   = initVelocity;
  velG_BCValueR   = initVelocity;
  for l=1:maxLevels
    NP = Levels{l}.NP;
    for ip=1:NP
      velP(ip,l)    = initVelocity;
      initPos(ip,l) = xp(ip,l);
    end
  end
end

if strcmp(problem_type, 'compaction')
  initVelocity    = 0;
  waveTansitTime  = bar_length/speedSound
  t_final          = waveTansitTime * 45;
  numBCs          = 1;
  delta_0         = 50;
  velG_BCValueL   = initVelocity;
  velG_BCValueR   = initVelocity;
  titleStr(1) = {'Quasi-Static Compaction Problem'};
  
end

if strcmp(problem_type, 'mms')
  initVelocity    = 0;
  waveTansitTime  = bar_length/speedSound
  t_initial       = 0.0 
  t_final         = waveTansitTime * 1.0;
  numBCs          = 1;
  delta_0         = 0;
  velG_BCValueL   = initVelocity;
  velG_BCValueR   = initVelocity;
  titleStr(1) = {'MMS Problem'};
  
  xp_initial = zeros(NP_max, maxLevels);
  
  for l=1:maxLevels
    xp_initial(:,l) = xp(:,l);
    [Fp(:,l)]      = mms.deformationGradient(xp_initial(:,l), t_initial, NP,speedSound, bar_length);
    [dp(:,l)]      = mms.displacement(       xp_initial(:,l), t_initial, NP, speedSound, bar_length);
    [velP(:,l)]    = mms.velocity(           xp_initial(:,l), t_initial, NP, speedSound, bar_length);
    [stressP(:,l)] = computeStress(E,Fp,NP);

    lp(:,l)  = lp(:,l) .* Fp(:,l);
    vol(:,l) = vol(:,l) .* Fp(:,l);
    xp(:,l)  =  xp_initial(:,l) + dp(:,l);
  end
end

%__________________________________
titleStr(2) = {sprintf('Computational Domain 0,%g, MPM bar %g,%g',domain,bar_min, bar_max)};
titleStr(3) = {sprintf('%s, PPC: %g',interpolation, PPC)};
%titleStr(4)={'Variable Resolution, Center Region refinement ratio: 2'}
titleStr(4) ={sprintf('Constant resolution, #cells %g', Limits.NN_allLevels)};


%plot initial conditions
if(plotSwitch == 1)
  plotResults(titleStr, t_initial, tstep, xp, dp, massP, Fp, velP, stressP, nodePos, velG, massG, momG, Limits, Levels)
end
fprintf('t_final: %g, interpolator: %s, NN: %g, NP: %g dx_min: %g \n',t_final,interpolation, Limits.NN_allLevels,Limits.NP_allLevels,dx_min);
input('hit return')

fn = sprintf('initialConditions.dat');
fid = fopen(fn, 'w');
fprintf(fid,'#p, xp, velP, Fp, stressP accl_extForceP\n');
for ip=1:NP
  fprintf(fid,'%g %16.15E %16.15E %16.15E %16.15E %16.15E\n',ip, xp(ip),velP(ip),Fp(ip), stressP(ip), accl_extForceP(ip));
end
fclose(fid);

%==========================================================================
% Main timstep loop
t = t_initial;
while t<t_final && tstep < max_tstep

  % compute the timestep
  dt = double(BigNum);
  for l=1:maxLevels
    L = Levels{l};
    for ip=1:L.NP
      dt = min(dt, CFL*dx_min/(speedSound + abs(velP(ip,l) ) ) );
    end
  end

  tstep = tstep + 1;
  t = t + dt;
  if (mod(tstep,20) == 0)
    fprintf('timestep %g, dt = %g, time %g \n',tstep, dt, t)
  end
  
  % initialize arrays to be zero
   for l=1:maxLevels   
    L = Levels{l};
          
    for ig=1:L.NN
      massG(ig,l)     =1.e-100;
      velG(ig,l)      =0.;
      vel_nobc_G(ig,l)=0.;
      accl_G(ig,l)    =0.;
      extForceG(ig,l) =0.;
      intForceG(ig,l) =0.;
    end
  end
  
  % compute the problem specific external force acceleration.
  %[accl_extForceP, delta] = ExternalForceAccl(problem_type, delta_0, bodyForce, Material, xp, xp_initial, t, tstep, NP, L1_dx, bar_length);
    
  %__________________________________
  % project particle data to grid  
  for l=1:maxLevels   
    L = Levels{l};
    
    for ip=1:L.NP

      [nodes,Ss]=sf.findNodesAndWeights_linear(xp(ip,l), lp(ip,l), L.nPatches, L.Patches, nodePos(:,l), Lx(:,:,l));
      for ig=1:NSFN
        massG(nodes(ig),l)     = massG(nodes(ig),l)     + massP(ip,l) * Ss(ig);
        velG(nodes(ig),l)      = velG(nodes(ig),l)      + massP(ip,l) * velP(ip,l) * Ss(ig);
        extForceG(nodes(ig),l) = extForceG(nodes(ig),l) + massP(ip,l) * accl_extForceP(ip,l) * Ss(ig); 

        % debugging__________________________________
        if(1)
          fprintf( 'L-%g  ip: %g xp: %g, nodes: %g, node_pos: %g massG: %g, massP: %g, Ss: %g,  prod: %g \n', l, ip, xp(ip,l), nodes(ig), nodePos(nodes(ig),l), massG(nodes(ig),l), massP(ip,l), Ss(ig), massP(ip,l) * Ss(ig) );
          fprintf( '\t velG:      %g,  velP:       %g,  prod: %g \n', velG(nodes(ig),l), velP(ip,l), (massP(ip,l) * velP(ip,l) * Ss(ig) ) );
          fprintf( '\t extForceG: %g,  accl_extForceP:  %g,  prod: %g \n', extForceG(nodes(ig),l), accl_extForceP(ip,l), accl_extForceP(ip,l) * Ss(ig) );
        end 
        % debugging__________________________________

      end
    end
  end

  % normalize by the mass
  velG = velG./massG;
  vel_nobc_G = velG;

  % set velocity BC
  for ibc=1:length(BCNodeL)
    velG(BCNodeL(ibc)) = velG_BCValueL;
    velG(BCNodeR(ibc)) = velG_BCValueR;
  end

  
  %compute internal force
  for l=1:maxLevels   
    L = Levels{l};
    
    for ip=1:L.NP
      [nodes,Gs,dx]=sf.findNodesAndWeightGradients_linear(xp(ip), lp(ip), nRegions, Regions, nodePos,Lx);
      for ig=1:NSFN
        intForceG(nodes(ig)) = intForceG(nodes(ig)) - Gs(ig) * stressP(ip) * vol(ip);
      end
    end
  end

  %__________________________________
  %compute the acceleration and new velocity on the grid
  accl_G    =(intForceG + extForceG)./massG;
  vel_new_G = velG + accl_G.*dt;
  

  %set velocity BC
  for ibc=1:length(BCNodeL)
    vel_new_G(BCNodeL(ibc)) = velG_BCValueL;
    vel_new_G(BCNodeR(ibc)) = velG_BCValueR;
  end

  momG = massG .* vel_new_G;

  %__________________________________
  % compute the acceleration on the grid
  for ig=1:NN
    accl_G(ig)  = (vel_new_G(ig) - vel_nobc_G(ig))/dt;
  end
  
  
  %set acceleration BC
  for ibc=1:length(BCNodeL)
    accl_G(BCNodeL(ibc)) = 0.0;
    accl_G(BCNodeR(ibc)) = 0.0;
  end
 
  
  %compute particle stress
  [Fp, dF,vol,lp] = computeDeformationGradient(xp,lp,dt,vel_new_G,Fp,NP, nRegions, Regions, nodePos,Lx);
  [stressP]       = computeStress(E,Fp,NP);
  
  %__________________________________
  %project changes back to particles
  tmp = zeros(NP,1);
  for ip=1:NP
    [nodes,Ss]=sf.findNodesAndWeights_linear(xp(ip), lp(ip), nRegions, Regions, nodePos, Lx);
    dvelP = 0.;
    dxp   = 0.;
    
    for ig=1:NSFN
      dvelP = dvelP + accl_G(nodes(ig))    * dt * Ss(ig);
      dxp   = dxp   + vel_new_G(nodes(ig)) * dt * Ss(ig);
    end

    velP(ip) = velP(ip) + dvelP;
    xp(ip)   = xp(ip) + dxp; 
    dp(ip)   = dp(ip) + dxp;
    tmp(ip)  = tmp(ip) + dxp;
  end
  
 
  
  DX_tip(tstep)=dp(NP);
  T=t; %-dt;

  %__________________________________
  % compute kinetic, strain and total energy
  KE(tstep)=0.;
  SE(tstep)=0.;
  totalMom(tstep) = 0;
  
  for ip=1:NP
    totalMom(tstep) = totalMom(tstep) + massP(ip) * velP(ip);
    KE(tstep) = KE(tstep) + .5*massP(ip) * velP(ip) * velP(ip);
    SE(tstep) = SE(tstep) + .5*stressP(ip) * (Fp(ip)-1.) * vol(ip);
    TE(tstep) = KE(tstep) + SE(tstep);
  end

  %__________________________________
  % Place data into structures
  G.nodePos   = nodePos;  % Grid based Variables
  
  P.velP      = velP;     % particle variables
  P.Fp        = Fp;
  P.xp        = xp;
  P.dp        = dp;
  P.extForceP = accl_extForceP;
  
  OV.speedSound = speedSound;    % Other Variables
  OV.NP         = NP;
  OV.E          = E;
  OV.t          = t;
  OV.tstep      = tstep;
  OV.bar_length = bar_length;
  
  %__________________________________
  % compute the exact solutions
  if strcmp(problem_type, 'impulsiveBar')
    if(T <= period/2.)
      Exact_tip(tstep) = M*T;
    else
      Exact_tip(tstep) = 4.*D-M*T;
    end
  end
  
  if strcmp(problem_type, 'oscillator')
    Exact_tip(tstep) = Amp*sin(2. * 3.14159 * T/period);
  end
  
  if strcmp(problem_type, 'advectBlock')
    Exact_tip(tstep) = DX_tip(tstep);
    
    pos_error  = 0;          % position error
    for ip=1:NP
      exact_pos = (initPos(ip) + t * initVelocity);
      pos_error = pos_error +  xp(ip) - exact_pos;
      %fprintf('xp: %16.15f  exact: %16.15f error %16.15f \n',xp(ip), exact_pos, xp(ip) - exact_pos)
    end
    %fprintf('sum position error %E \n',sum(pos_error))
  end
  
  if( strcmp(problem_type, 'compaction') && (mod(tstep,plotInterval) == 0) && (plotSwitch == 1) )
    term1 = (2.0 * density * bodyForce)/E;
    for ip=1:NP
      term2 = term1 * (delta - xp(ip));
      stressExact(ip) = E *  ( sqrt( term2 + 1.0 ) - 1.0);
    end
    
    figure(2)
    set(2,'position',[1000,100,700,700]);

    plot(xp,stressP,'rd', xp, stressExact, 'b');
    axis([0 50 -10000 0])

    title(titleStr)
    legend('Simulation','Exact')
    xlabel('Position');
    ylabel('Particle stress');

    f_name = sprintf('%g.2.ppm',tstep-1);
    F = getframe(gcf);
    [X,map] = frame2im(F);
    imwrite(X,f_name);

    % compute L2Norm
    d = abs(stressP - stressExact);
    L2_norm = sqrt( sum(d.^2)/length(stressP) )
  end
  
  
  if (strcmp(problem_type, 'mms'))
    [L2_norm, maxError] = mms.plotResults(titleStr, plotSwitch, plotInterval, xp_initial, OV, P, G);
  end
  
  
  TIME(tstep)=t;
  
  %__________________________________
  % plot intantaneous solution
  if (mod(tstep,plotInterval) == 0) && (plotSwitch == 1)
    plotResults(titleStr, t, tstep, xp, dp, massP, Fp, velP, stressP, nodePos, velG, massG, momG)
    %input('hit return');
  end
  
  %__________________________________
  % compute on the errors
  totalEng_err(tstep)   =TE(1)-TE(tstep);                      % total energy
  totalMom_err(tstep)   =totalMom(1) - totalMom(tstep);        % total momentum
  tipDeflect_err(tstep) =abs(DX_tip(tstep)-Exact_tip(tstep));  % tip deflection  
  
  %__________________________________
  % bulletproofing
  % particles can't leave the domain
  for ip=1:NP
    if(xp(ip) >= domain) 
      t = t_final;
      fprintf('\nparticle(%g) position is outside the domain: %g \n',ip,xp(ip))
      fprintf('now exiting the time integration loop\n\n') 
    end
  end
end  % main loop

totalEng_err(tstep)
totalMom_err(tstep)
tipDeflect_err(tstep)


%==========================================================================
%  plot the results
%plotFinalResults(TIME, DX_tip, Exact_tip, TE, problem_type, PPC, NN)

%__________________________________
%  write the results out to files
% particle data

if (writeData == 1)
  fname1 = sprintf('particle_NN_%g_PPC_%g.dat',NN, PPC);
  fid = fopen(fname1, 'w');
  fprintf(fid,'#%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
  fprintf(fid,'#p, xp, velP, Fp, stressP, time\n');
  for ip=1:NP
    fprintf(fid,'%g %16.15E %16.15E %16.15E %16.15E %16.15E\n',ip, xp(ip),velP(ip),Fp(ip), stressP(ip), t);
  end
  fclose(fid);

  % grid data
  fname2 = sprintf('grid_NN_%g_PPC_%g.dat',NN, PPC);
  fid = fopen(fname2, 'w');
  fprintf(fid,'#%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
  fprintf(fid,'#node, xG, massG, velG, extForceG, intForceG, accl_G\n');
  for ig=1:NN
    fprintf(fid,'%g, %16.15f, %16.15f, %16.15f, %16.15f, %16.15f %16.15f\n',ig, nodePos(ig), massG(ig), velG(ig), extForceG(ig), intForceG(ig), accl_G(ig));
  end
  fclose(fid);

  % conserved quantities
  fname3 = sprintf('conserved_NN_%g_PPC_%g.dat',NN, PPC);
  fid = fopen(fname3, 'w');
  fprintf(fid,'#%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
  fprintf(fid,'timesetep, KE, SE, TE, totalMom\n');
  for t=1:length(TIME)
    fprintf(fid,'%16.15f, %16.15f, %16.15f, %16.15f, %16.15f\n',TIME(t), KE(t), SE(t), TE(t), totalMom(t));
  end
  fclose(fid);

  % errors
  fname4 = sprintf('errors_NN_%g_PPC_%g.dat',NN, PPC);
  fid = fopen(fname4, 'w');
  fprintf(fid,'#%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
  fprintf(fid,'#timesetep, totalEng_err, totalMom_err, tipDeflect_err\n');
  for t=1:length(TIME)
    fprintf(fid,'%16.15f, %16.15f, %16.15f, %16.15f\n',TIME(t), totalEng_err(t), totalMom_err(t), tipDeflect_err(t));
  end
  fclose(fid);

  fprintf(' Writing out the data files \n\t %s \n\t %s \n\t %s \n\t %s \n',fname1, fname2, fname3,fname4);
end


end
%______________________________________________________________________
% functions
%______________________________________________________________________
function [Fp, dF, vol, lp] = computeDeformationGradient(xp,lp,dt,velG,Fp,NP, nRegions, Regions, nodePos,Lx)
  global NSFN;
  global sf;
  
  for ip=1:NP
    [nodes,Gs,dx]  = sf.findNodesAndWeightGradients_linear(xp(ip), lp(ip), nRegions, Regions, nodePos, Lx);
    [volP_0, lp_0] = sf.positionToVolP(xp(ip), nRegions, Regions);

    gUp=0.0;
    for ig=1:NSFN
      gUp = gUp + velG(nodes(ig)) * Gs(ig);
    end

    dF(ip)      = 1. + gUp * dt;
    Fp(ip)      = dF(ip) * Fp(ip);
    vol(ip)     = volP_0 * Fp(ip);
    lp(ip)      = lp_0 * Fp(ip);
  end
end


%__________________________________
function [stressP]=computeStress(E,Fp,NP)
                                                                                
  for ip=1:NP
%    stressP(ip) = E * (Fp(ip)-1.0);
    stressP(ip) = (E/2.0) * ( Fp(ip) - 1.0/Fp(ip) );        % hardwired for the mms test  see eq 50
  end
end




%__________________________________
function plotResults(titleStr,t, tstep, xp, dp, massP, Fp, velP, stressP, nodePos, velG, massG, momG, Limits, Levels)

    % plot SimulationState
  figure1 = figure(1)
  set(1,'position',[50,100,700,700]);
  levelColors = ['m','g','r','b','k'];
  
  subplot(4,1,1),plot(xp,velP,'rd');
  ylim([min(min(velP - 1e-3) )  max( max(velP + 1e-3))])
  
  xlabel('Particle Position');
  ylabel('Particle velocity');
  title(titleStr);
  hold on
  
  % draw vertical lines at each node 
  for L=1:Limits.maxLevels
    NN = Levels{L}.NN;
    for n=1:NN            
      x = nodePos(n,L);
      [pt1,pt2] = dsxy2figxy([x,x],ylim);
      annotation(figure1,'line',pt1,pt2,'Color',levelColors(L));
    end
  end
  hold off
  %axis([0 50 99 101] )

  subplot(4,1,2),plot(xp,Fp,'rd');
  %axis([0 50 0 2] )
  ylabel('Fp');

  subplot(4,1,3),plot(xp,stressP,'rd');
  %axis([0 50 -1 1] )
  ylabel('Particle stress');
  
  subplot(4,1,4),plot(xp,massP,'rd');
  %axis([0 50 -1 1] )
  ylabel('Particle mass');
  
  
if(0)
  subplot(6,1,4),plot(nodePos, velG,'bx');
  xlabel('NodePos');
  ylabel('grid Vel');
  %axis([0 50 0 101] )

  grad_velG = diff(velG);
  grad_velG(length(velG)) = 0;
  
  for n=2:length(velG)
    grad_velG(n) = grad_velG(n)/(nodePos(n) - nodePos(n-1) );
  end
  subplot(6,1,5),plot(nodePos, grad_velG,'bx');
  ylabel('grad velG');
  xlim([0,40])
  
  %subplot(6,1,5),plot(nodePos, massG,'bx');
  %ylabel('gridMass');
  %axis([0 50 0 1.1] )

  momG = velG .* massG;
  subplot(6,1,6),plot(nodePos, momG,'bx');
  ylabel('gridMom');
  %axis([0 50 0 101] )

  f_name = sprintf('%g.ppm',tstep-1);
  F = getframe(gcf);
  [X,map] = frame2im(F);
  imwrite(X,f_name)
  %input('hit return');
end
end

%__________________________________
function plotFinalResults(TIME, DX_tip, Exact_tip, TE, problem_type, PPC, NN)

  close all;
  set(gcf,'position',[100,100,900,900]);

  % tip displacement vs time
  subplot(3,1,1),plot(TIME,DX_tip,'bx');

  hold on;
  subplot(3,1,1),plot(TIME,Exact_tip,'r-');

  tmp = sprintf('%s, ppc: %g, NN: %g',problem_type, PPC, NN);
  title(tmp);
  legend('Simulation','Exact')
  xlabel('Time [sec]');
  ylabel('Tip Deflection')

  %total energy vs time
  subplot(3,1,2),plot(TIME,TE,'b-');
  ylabel('Total Energy');


  % compute error
  err=abs(DX_tip-Exact_tip);

  subplot(3,1,3),plot(TIME,err,'b-');
  ylabel('Abs(error)')
end


%__________________________________
function [accl_extForceP, delta] = ExternalForceAccl(problem_type, delta_0, bodyForce, Material, xp, xp_initial, t, tstep, NP, L1_dx, bar_length)

  density    = Material{1}.density;
  E          = Material{1}.E;
  speedSound = Material{1}.speedSound;
  
  delta = -9;
  
  if (strcmp(problem_type, 'compaction'))
    if(bodyForce > -200)
      bodyForce = -t * 100;
    end
    
    delta = delta_0 + (density*bodyForce/(2.0 * E) ) * (delta_0 * delta_0);  

    displacement = delta - delta_0;
    W = density * abs(bodyForce) * delta_0; 
    
    if (mod(tstep,100) == 0)
      fprintf('Bodyforce: %g displacement:%g, W: %g\n',bodyForce, displacement/L1_dx, W);                                             
    end
    for ip=1:NP                                                                       
       accl_extForceP(ip) = bodyForce;                                                      
    end                                                                               
  end
  
  if (strcmp(problem_type, 'mms'))
    [mms] = MMS;                % load mms functions
    [accl_extForceP] = mms.accl_bodyForce(xp_initial,t, NP, speedSound, bar_length);
  end
end
