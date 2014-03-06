% Reference:  Jin Ma, Honbing Lu,and Ranga Komanduri, "Structured Mesh
% Refinement in Generalized Interpolation Material Point (GIMP) Method
% for Simulation of Dynamic Problems," CMES, vol. 12, no.3, pp. 213-227, 2006
%______________________________________________________________________
                                                         
   
   

function [L2_norm,maxError,NN,NP]=ml_amrmpm(problem_type,CFL,NCells)
unix('/bin/rm ml_1 ml_2 ml_0')
close all
intwarning on
addpath([docroot '/techdoc/creating_plots/examples'])   % so you can use dsxy2figxy()


global d_debugging;
global PPC;
global sf;
global gf;
global NSFN;               % number of shape function nodes
global dumpFrames;
d_debugging = problem_type;

[mms] = MMS;                      % load mms functions
[sf]  = shapeFunctions;           % load all the shape functions
[gf]  = gridFunctions;            % load all grid based function
[IF]  = initializationFunctions;   % load initialization functions
[pf]  = particleFunctions;        % load particle function

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
E       = 1.0e6;
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
L1         = 1;

bar_min     = 0.0;
bar_max     = 50.0;
bar_length  = bar_max - bar_min;

domain       = 52;
area         = 1.;
plotSwitch   = 1;
plotInterval = 200;
dumpFrames   = 1;
writeData    = 0;
max_tstep    = 1000;

L1_dx       =domain/(NCells);

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

[Levels]                 = IF.findCFI_nodes(Levels, nodePos, Limits);

[Lx]                     = IF.initialize_Lx(nodePos, Levels, Limits);

[P.xp, Levels, Limits]     = IF.initialize_xp(nodePos, interpolation, PPC, bar_min, bar_max, Limits, Levels);

[Limits]                 = IF.NN_NP_allLevels(Levels, Limits);

% define what the interpolation dx for each level
% set the number of particles in the set
for l=1:Limits.maxLevels
  P.interpolation_dx(l) = Levels{l}.dx;
  P.NP(l) = Levels{l}.NP;
end

% set the computational domain tag
P.CompDomain        = 'nonExtraCells';

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
      % particle arrays
P.vol       = NaN(NP_max, maxLevels);
P.lp        = NaN(NP_max, maxLevels);
P.massP     = NaN(NP_max, maxLevels);
P.velP      = NaN(NP_max, maxLevels);
P.dp        = NaN(NP_max, maxLevels);
P.Fp        = NaN(NP_max, maxLevels);
P.dF        = NaN(NP_max, maxLevels);

P.stressP        = zeros(NP_max, maxLevels);  % must be full of zeros
P.accl_extForceP = zeros(NP_max,maxLevels);

      % node based arrays
nodes     = zeros(1,NSFN);
Gs        = zeros(1,NSFN);
Ss        = zeros(1,NSFN);

massG     = NaN(NN_max,  maxLevels);
momG      = NaN(NN_max,  maxLevels);
velG      = NaN(NN_max,  maxLevels);
vel_nobc_G= NaN(NN_max,  maxLevels);
accl_G    = NaN(NN_max,  maxLevels);
extForceG = zeros(NN_max,  maxLevels);
intForceG = NaN(NN_max,  maxLevels);


KE        = zeros(BigNum,1);
SE        = zeros(BigNum,1);
TE        = zeros(BigNum,1);
totalMom  = zeros(BigNum,1);
Exact_tip = zeros(BigNum,1);
sumError  = zeros(BigNum,1);
DX_tip    = zeros(BigNum,1);

totalEng_err   = zeros(BigNum,1);
totalMom_err   = zeros(BigNum,1);
tipDeflect_err = zeros(BigNum,1);

input('hit return')

%__________________________________
% initialize other particle variables
for l=1:maxLevels
   L = Levels{l};
   
  for ip=1: L.NP
    [volP_0, lp_0] = sf.positionToVolP(P.xp(ip,l), L.dx, L.Patches);

    P.vol(ip,l)   = volP_0;
    P.massP(ip,l) = volP_0*density;
    P.lp(ip,l)    = lp_0;
    P.Fp(ip,l)    = 1.;                     % total deformation
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
  P.massP(NP)       = Mass;                    % last particle masss
  P.velP(NP)        = v0;                      % last particle velocity
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
  
  ipp = 0;
  for l=1:maxLevels
    L = Levels{l};
    for ip=1:L.NP
      ipp = ipp + 1;
      P.velP(ip,l)    = initVelocity;
      initPos(ipp)  = xp(ip,l);
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
  
  xp_initial = zeros(NP_max,1);
  xp_initial = P.xp;
  
  for l=1:maxLevels
    L = Levels{l};
    for ip=1:L.NP
      P.velP(ip,l)    = initVelocity;
    end
  end
  
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
    xp_initial(:,l) = P.xp(:,l);
    [P.Fp(:,l)]      = mms.deformationGradient(xp_initial(:,l), t_initial, NP,speedSound, bar_length);
    [P.dp(:,l)]      = mms.displacement(       xp_initial(:,l), t_initial, NP, speedSound, bar_length);
    [P.velP(:,l)]    = mms.velocity(           xp_initial(:,l), t_initial, NP, speedSound, bar_length);
    [P.stressP(:,l)] = computeStress(E,Fp,NP);

    P.lp(:,l)  = P.lp(:,l) .* P.Fp(:,l);
    P.vol(:,l) = P.vol(:,l) .* P.Fp(:,l);
    P.xp(:,l)  = xp_initial(:,l) + P.dp(:,l);
  end
end


%__________________________________
titleStr(2) = {sprintf('Computational Domain 0,%g, MPM bar %g,%g',domain,bar_min, bar_max)};
titleStr(3) = {sprintf('%s, PPC: %g',interpolation, PPC)};
%titleStr(4)={'Variable Resolution, Center Region refinement ratio: 2'}
titleStr(4) ={sprintf('Multi-Levels, #cells %g', Limits.NN_allLevels)};


%plot initial conditions
if(plotSwitch == 1)
  plotResults(titleStr, t_initial, tstep, P, nodePos, velG, massG, momG,extForceG,intForceG, Limits, Levels)
end
fprintf('t_final: %g, interpolator: %s, NN: %g, NP: %g dx_min: %g \n',t_final,interpolation, Limits.NN_allLevels,Limits.NP_allLevels,dx_min);
input('hit return')

fn = sprintf('initialConditions.dat');
fid = fopen(fn, 'w');
fprintf(fid,'level, #p, xp, velP, Fp, stressP accl_extForceP\n');


for l=1:maxLevels
  L = Levels{l};  
  for ip=1:L.NP
    fprintf(fid,'%g, %g %16.15E %16.15E %16.15E %16.15E %16.15E\n',l,ip, P.xp(ip,l),P.velP(ip,l),P.Fp(ip,l), P.stressP(ip,l), P.accl_extForceP(ip,l));
  end
end
fclose(fid);
%==========================================================================
% Main timstep loop
t = t_initial;
while t<t_final && tstep < max_tstep

  fid0 = fopen('ml_0','a');
  fid1 = fopen('ml_1','a');
  fid2 = fopen('ml_2','a');
  fprintf(fid0,'__________________________________\n\n');
  fprintf(fid1,'__________________________________\n\n');
  fprintf(fid2,'__________________________________\n\n');
  fclose(fid0);
  fclose(fid1);
  fclose(fid2);

  % compute the timestep
  dt = double(BigNum);
  for l=1:maxLevels
    L = Levels{l};
    for ip=1:L.NP
      dt = min(dt, CFL*dx_min/(speedSound + abs(P.velP(ip,l) ) ) );
    end
  end

  tstep = tstep + 1;
  t = t + dt;
  if (mod(tstep,20) == 0)
    fprintf('timestep %g, dt = %g, time %g \n',tstep, dt, t)
  end
  
  % initialize grid arrays to be zero
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
  % do not compute on the extra cell particles
  for l=1:maxLevels
     L = Levels{l};
    [P.accl_extForceP(:,l), delta,bodyForce] = ExternalForceAccl(problem_type, delta_0, bodyForce, Material, P.xp(:,l), xp_initial(:,l), t, tstep, L.NP, L1_dx, bar_length);
  end  
  
  % determine which particle are "extra cell particles"
  % create th extra cell particle struct
   %[P]        = pf.createParticleStruct(xp, massP, velP, stressP, vol, lp, dp, Fp);
  [pID]      = pf.findExtraCellParticles(P, Levels, Limits, nodePos, interpolation);
  [EC_Pdata] = pf.createEC_particleStruct(P, pID, Levels, Limits);
  


  % create the struct that contains the two particle structs
  PSets{1} = P;
  PSets{2} = EC_Pdata;
  
  %__________________________________
  % project particle data to grid for each particle set
  for pset = 1:length(PSets)
    ps = PSets{pset};
    
    [xp, massP, velP, accl_extForceP, lp]= pf.getCopy(ps,'xp','massP','velP','accl_extForceP', 'lp');
    compDomain = ps.CompDomain;
    
    nsfn = NSFN;
    if( strcmp(compDomain,'ExtraCells'))
      nsfn = 1;
    end
    
    for l=1:maxLevels
      L = Levels{l};
      NP = ps.NP(l);
      
      for ip=1:NP

        [nodes,Ss]=sf.findNodesAndWeights_linear(xp(ip,l), lp(ip,l), L.CFI_nodes, nodePos(:,l), Lx(:,:,l), compDomain, nsfn);
        for ig=1:nsfn
          massG(nodes(ig),l)     = massG(nodes(ig),l)     + massP(ip,l) * Ss(ig);
          velG(nodes(ig),l)      = velG(nodes(ig),l)      + massP(ip,l) * velP(ip,l) * Ss(ig);
          extForceG(nodes(ig),l) = extForceG(nodes(ig),l) + massP(ip,l) * accl_extForceP(ip,l) * Ss(ig); 

          % debugging__________________________________
          %if( any(nodes(ig) == L.CFI_nodes(:)) )
          n = nodePos(nodes(ig),l);
          %if( ( (n >= 17.333 && n <= 17.34) || (n >= 34.6 && n <= 34.7) )  && l == 1)
          if( (n >= 34.6 && n <= 34.7)   && l == 2)
            fid = fopen('ml_0','a');
            fprintf( fid,'L-%g  ip: %g xp: %g, nodes: %g, node_pos: %g massG: %g, massP: %g, Ss: %g,  prod: %g \n', l, ip, xp(ip,l), nodes(ig), nodePos(nodes(ig),l), massG(nodes(ig),l), massP(ip,l), Ss(ig), massP(ip,l) * Ss(ig) );
            fprintf( fid,'\t velG:      %g,  velP:       %g,  prod: %g \n', velG(nodes(ig),l), velP(ip,l), (massP(ip,l) * velP(ip,l) * Ss(ig) ) );
            fprintf( fid,'\t extForceG: %g,  accl_extForceP:  %g,  prod: %g \n', extForceG(nodes(ig),l), accl_extForceP(ip,l), accl_extForceP(ip,l) * Ss(ig) );
            fclose(fid);
          end 
          % debugging__________________________________

        end
      end  %particles
    end  % levels
  end  % particle set
  
  
  %__________________________________
  % adjust the nodal values at the CFI  
  %[massG, velG, extForceG] = adjustCFI_Nodes(Levels, Limits,massG, velG, extForceG );
  
  
  % normalize by the mass
  velG = velG./massG;
  vel_nobc_G = velG;

  % set velocity BC on L1
  for ibc=1:length(BCNodeL)
    velG(BCNodeL(ibc),L1) = velG_BCValueL;
    velG(BCNodeR(ibc),L1) = velG_BCValueR;
  end

  %compute internal force for each particle set
  % with/without the extra cells
  for pset = 1:length(PSets)
    ps = PSets{pset};
    
    [xp, lp, stressP, vol]= pf.getCopy(ps,'xp','lp','stressP','vol');

    compDomain = ps.CompDomain;
    
    nsfn = NSFN;
    if( strcmp(compDomain,'ExtraCells'))
      nsfn = 1;
    end
    
    for l=1:maxLevels   
      L = Levels{l};

      dx = ps.interpolation_dx(l);
      
      for ip=1:ps.NP(l)
        [nodes,Gs]=sf.findNodesAndWeightGradients_linear(xp(ip,l), lp(ip,l), dx, L.CFI_nodes, nodePos(:,l), Lx(:,:,l), compDomain);
        
        for ig=1:nsfn
          source =  Gs(ig) * stressP(ip,l) * vol(ip,l);
          intForceG(nodes(ig),l) = intForceG(nodes(ig),l) - source;
         
          % debugging__________________________________
          %if( any(nodes(ig) == L.CFI_nodes(:)) )
          n = nodePos(nodes(ig),l);
          %if( ( (n >= 17.333 && n <= 17.34) || (n >= 34.6 && n <= 34.7) )  && l == 1)
          if( (n >= 34.6 && n <= 34.7)   && l == 2)
            fid = fopen('ml_1','a');
            fprintf( fid, 'L-%g  compDomain:%s nodePos: %g, ip: %g  internalForceG: %g source: %g, stressP: %g vol: %g Gs: %g \n', l, compDomain, n, ip, intForceG(nodes(ig),l), source, stressP(ip,l), vol(ip,l), Gs(ig)  );
            fclose(fid);
          end
          % debugging__________________________________
        end
      end
    end  % level
  end  % particle Set

 
 % [intForceG] = adjustCFI_Nodes(Levels, Limits,intForceG );

  %__________________________________
  %compute the acceleration and new velocity on the grid
  accl_G    =(intForceG + extForceG)./massG;
  vel_new_G = velG + accl_G.*dt;
  

  %set velocity BC on L1
  for ibc=1:length(BCNodeL)
    vel_new_G(BCNodeL(ibc),L1) = velG_BCValueL;
    vel_new_G(BCNodeR(ibc),L1) = velG_BCValueR;
  end

  momG = massG .* vel_new_G;

  %__________________________________
  % compute the acceleration on the grid
  for l=1:maxLevels   
    L = Levels{l};
          
    for ig=1:L.NN
      accl_G(ig,l)  = (vel_new_G(ig,l) - vel_nobc_G(ig,l))/dt;
    end
  end
  
  %set acceleration BC on L1
  for ibc=1:length(BCNodeL)
    accl_G(BCNodeL(ibc)) = 0.0;
    accl_G(BCNodeR(ibc)) = 0.0;
  end
  
  %compute particle stress
  for l=1:maxLevels 
    L = Levels{l};
        
    if(L.NP > 0)     
      [P.Fp(:,l), P.dF(:,l), P.vol(:,l), P.lp(:,l)] = computeDeformationGradient(P.xp(:,l), P.lp(:,l), P.Fp(:,l), dt, vel_new_G(:,l), L.NP, L.dx, L.Patches, nodePos(:,l), Lx(:,:,l));
      [P.stressP(:,l)]                      = computeStress(E, P.Fp(:,l), L.NP);
    end
  end
  
  %__________________________________
  %project changes back to particles
  for l=1:maxLevels 
    L = Levels{l};
    
    compDomain = 'noExtraCells';

    for ip=1:L.NP
      [nodes,Ss]=sf.findNodesAndWeights_linear(P.xp(ip,l), P.lp(ip,l), L.CFI_nodes, nodePos(:,l), Lx(:,:,l), compDomain, NSFN);
      dvelP = 0.;
      dxp   = 0.;

      for ig=1:NSFN
        dvelP = dvelP + accl_G(nodes(ig),l)    * dt * Ss(ig);
        dxp   = dxp   + vel_new_G(nodes(ig),l) * dt * Ss(ig);
      end

      P.velP(ip,l) = P.velP(ip,l) + dvelP;
      P.xp(ip,l)   = P.xp(ip,l) + dxp; 
      P.dp(ip,l)   = P.dp(ip,l) + dxp;
    end
  end

  
  %__________________________________
  % relocate particles between levels
  [P.xp,P.massP,P.velP,P.vol,P.Fp,P.lp, Levels]=relocateParticles(P.xp,P.massP,P.velP,P.vol,P.Fp,P.lp,Levels,Limits);
  
  %define number of particles in the main particle set
  for l=1:Limits.maxLevels   
    L = Levels{l};
    P.NP(l) = L.NP;
  end
  
  
  %__________________________________
  % find the tip displacement
  for l=1:maxLevels 
    [tmp,ip]        = max(P.xp(:,l) );
    tipLocation     = P.xp(ip,l);
    tipDisplacement = P.dp(ip,l);
  end
  
  DX_tip(tstep)=tipDisplacement;
  T=t; %-dt;

  %__________________________________
  % compute kinetic, strain and total energy over all levels
  KE(tstep) = 0.0;
  SE(tstep) = 0.0;
  totalMom(tstep) = 0.0;
  
  for l=1:maxLevels 
    L = Levels{l};
    for ip=1:L.NP
      totalMom(tstep) = totalMom(tstep) + P.massP(ip,l) * P.velP(ip,l);
      KE(tstep) = KE(tstep) + .5*P.massP(ip,l) * P.velP(ip,l) * P.velP(ip,l);
      SE(tstep) = SE(tstep) + .5*P.stressP(ip,l) * (P.Fp(ip,l)-1.) * P.vol(ip,l);
      TE(tstep) = KE(tstep) + SE(tstep);
    end
  end
  
  %__________________________________
  % Place data into structures
  G.nodePos   = nodePos;  % Grid based Variables
  
  OV.speedSound = speedSound;    % Other Variables
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
    % remove NaN's and convert then to 1D arrays
    xp_clean    = xp(isfinite(xp));
  
    sumExactPos = sum(initPos + t * initVelocity );
    sumCurPos = sum(xp_clean);
    fprintf('sum position error %E \n',sumExactPos - sumCurPos);
    sumError(tstep) = sumExactPos - sumCurPos;
  end
  
  if( strcmp(problem_type, 'compaction') && (mod(tstep,plotInterval) == 0) && (plotSwitch == 1) )
    %convert the multi-level arrays into a 1D arrays
    [xp_clean, stressP_clean] = ML_to_1D( Levels, Limits, P.xp, P.stressP);
    NP = length(xp_clean);
    
    term1 = (2.0 * density * bodyForce)/E;
    for ip=1:NP
      term2 = term1 * (delta - xp_clean(ip));
      stressExact(ip) = E *  ( sqrt( term2 + 1.0 ) - 1.0);
    end
    
    fig3 = sfigure(3);
    set(fig3,'position',[1000,100,700,700]);

    plot(xp_clean,stressP_clean,'rd', xp_clean, stressExact, 'b');
    
    
    axis([0 50 -10000 0])

    title(titleStr)
    legend('Simulation','Exact')
    xlabel('Position');
    ylabel('Particle stress');
    
    if(dumpFrames)
      f_name = sprintf('%g.3.ppm',tstep-1);
      F = getframe(fig3);

      [X,map] = frame2im(F);
      imwrite(X,f_name);
    end
    % compute L2Norm
    d = abs(stressP_clean - stressExact);
    L2_norm = sqrt( sum(d.^2)/length(stressP_clean) )
  end
  
  
  if (strcmp(problem_type, 'mms'))
    [L2_norm, maxError] = mms.plotResults(titleStr, plotSwitch, plotInterval, xp_initial, OV, P, G);
  end
  
  
  TIME(tstep)=t;
  
  %__________________________________
  % plot intantaneous solution
  if (mod(tstep,plotInterval) == 0) && (plotSwitch == 1)
    plotResults(titleStr, t, tstep, P, nodePos, velG, massG, momG,extForceG,intForceG,Limits, Levels)
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
  for l=1:maxLevels 
    L = Levels{l};
    
    rougeParticle = find(P.xp(:,l) >= L.max | P.xp(:,l) < L.min);
    
    if( ~ isempty(rougeParticle)) 
      fprintf('\n L-%g, particle(%g) position is outside the domain [%g, %g]: %g \n',l,ip,L.min, L.max,xp(ip,l))
      fprintf('now exiting the time integration loop\n\n');
      return; 
    end

  end
end  % main loop

totalEng_err(tstep);
totalMom_err(tstep);
tipDeflect_err(tstep);


%==========================================================================
%  plot the results
%plotFinalResults(TIME, DX_tip, Exact_tip, TE, problem_type, PPC, NN)

%__________________________________
%  write the results out to files
% particle data

if (writeData == 1)
  NN = Limits.NN_max
  NP = Limits.NP_max
  fname1 = sprintf('particle_NN_%g_PPC_%g.dat',NN, PPC);
  fid = fopen(fname1, 'w');
  fprintf(fid,'#%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
  fprintf(fid,'#p, xp, velP, Fp, stressP, time\n');
  for ip=1:NP
    fprintf(fid,'%g %16.15E %16.15E %16.15E %16.15E %16.15E\n',ip, P.xp(ip),P.velP(ip),P.Fp(ip), P.stressP(ip), t);
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
  fprintf(fid,'#timesetep, totalEng_err, totalMom_err, tipDeflect_err, sumError\n');
  for t=1:length(TIME)
    fprintf(fid,'%16.15E, %16.15E, %16.15E, %16.15E, %16.15E\n',TIME(t), totalEng_err(t), totalMom_err(t), tipDeflect_err(t),sumError(t) );
  end
  fclose(fid);

  fprintf(' Writing out the data files \n\t %s \n\t %s \n\t %s \n\t %s \n',fname1, fname2, fname3,fname4);
end


end
%______________________________________________________________________
% functions
%______________________________________________________________________
function [Fp, dF, vol, lp] = computeDeformationGradient(xp, lp, Fp, dt ,velG, NP, dx, Regions, nodePos, Lx)
  global NSFN;
  global sf;
  
  nn  = length(lp);
  vol = NaN(nn,1);     % you must declare arrays that are not passed in.
  dF  = NaN(nn,1);
  
  notUsed   = NaN(1); 
  CFI_nodes = NaN(1);
  compDomain = 'nonExtraCells';
  
  for ip=1:NP
    [nodes,Gs]  = sf.findNodesAndWeightGradients_linear(xp(ip), lp(ip), dx, CFI_nodes, nodePos, Lx, compDomain);

    [volP_0, lp_0] = sf.positionToVolP(xp(ip), dx, Regions);

    gUp=0.0;
    for ig=1:NSFN
      gUp = gUp + velG(nodes(ig)) * Gs(ig);
      
      if(1)
      %if(xp(ip) >= 17.333 && xp(ip) <= 18.45) 
       if(xp(ip) >=  34.156 && xp(ip) <= 35.17)
        fid = fopen('ml_2','a');
        fprintf(fid,'nodePos:%g xp:%g, gUp:%16.15E, velG:%16.15E Gs:%16.15E \n',nodePos(nodes(ig)),xp(ip), gUp, velG(nodes(ig)), Gs(ig));
        fclose(fid);
      end
      end
    end
 
    dF(ip,1)      = 1. + gUp * dt;
    Fp(ip,1)      = dF(ip) * Fp(ip);
    vol(ip,1)     = volP_0 * Fp(ip);
    lp(ip,1)      = lp_0 * Fp(ip);
    
    if(0)
    %if(xp(ip) >= 17.333 && xp(ip) <= 18.45) 
    if(xp(ip) >=  34.156 && xp(ip) <= 35.17)
      fid = fopen('ml_2','a');
      fprintf(fid,'xp:%g, gUp:%16.15E, Fp:%16.15E \n',xp(ip), gUp, Fp(ip));
      fclose(fid);
    end 
    end
    
  end
end


%__________________________________
function [stressP]=computeStress(E,Fp,NP)
  nn = length(Fp);
  stressP = zeros(nn,1);  % you must declare arrays that are not passed in.
                          % This array must be full of zeros
                                                                                  
  for ip=1:NP
    stressP(ip) = E * (Fp(ip)-1.0);
%    stressP(ip) = (E/2.0) * ( Fp(ip) - 1.0/Fp(ip) );        % hardwired for the mms test  see eq 50
  end
end

%__________________________________
function [varargout] = adjustCFI_Nodes( Levels, Limits, varargin)
  
  left  = 1;
  right = 2;
  
  for i = 1:length(varargin)    % loop over each input argument
    varG_new = varargin{i};
    varG     = varargin{i};
    varG_tmp = zeros(2,1);
    
    %Q_tmp(CFI_node) = sum(q(CFI_node,levels));
    for side=left:right
      for l=2:Limits.maxLevels
        cl = l -1;
        fineNode   = Levels{l}.CFI_nodes(side);
        coarseNode = Levels{cl}.CFI_nodes(side);
        varG_tmp(side) = varG_tmp(side) + varG(fineNode,l)  + varG(coarseNode,cl);

      end
    end

    %Q(CFI_node,AllLevels) = Q_tmp(CFI_node)
    for side=left:right
      for fl=2:Limits.maxLevels
        cl = fl -1;
        fineNode   = Levels{fl}.CFI_nodes(side);
        coarseNode = Levels{cl}.CFI_nodes(side);
        varG_new(fineNode,l)    = varG_tmp(side);
        varG_new(coarseNode,cl) = varG_tmp(side);
      end
    end

    
    varargout{i} = varG_new;
  end  % loop over input variables
end

%__________________________________
% relocate particles to the finest level
function [xp,massP,velP,vol,Fp,lp,Levels]=relocateParticles(xp,massP,velP,vol,Fp,lp,Levels,Limits)
  global gf;
  compDomain = 'withoutExtraCells';
  
  for l=1:Limits.maxLevels   
    L = Levels{l};
    
    for ip=1:L.NP   % has particle moved from coarse level to fine level
      moveToFinerLevel   = gf.hasFinerCell(xp(ip,l),l,Levels,Limits,compDomain);
      moveToCoarserLevel = gf.isOutsideLevel(xp(ip,l),l,Levels);

      if(moveToFinerLevel)
        newLevel = l +1;
        fprintf('L-%g, particle id: %g  moving particle to finer level %g, position: %g\n',l, ip, newLevel, xp(ip,l));
      end

      if(moveToCoarserLevel)
        newLevel = l -1; 
        fprintf('L-%g, particle id: %g  moving particle to coarser level %g position: %g\n',l, ip, newLevel, xp(ip,l));
      end
      
      if(moveToFinerLevel || moveToCoarserLevel)
      
        % Add particle state to the new level     
        newNP    = Levels{newLevel}.NP +1;  
        fprintf(' Relocating Particle, Lnew:%g, Lold:%g particle id_old:%g  id_new:%g \n',newLevel,l,ip,newNP);
        xp(newNP, newLevel)    = xp(ip,l);
        massP(newNP, newLevel) = massP(ip,l);  
        velP(newNP, newLevel)  = velP(ip,l);
        Fp(newNP, newLevel)    = Fp(ip,l);     
        lp(newNP, newLevel)    = lp(ip,l);     
        vol(newNP,newLevel)    = vol(ip,l);
        Levels{newLevel}.NP    = newNP;
        
        % delete particle from level
        xp(ip,l)    = NaN;
        massP(ip,l) = NaN;
        velP(ip,l)  = NaN;
        Fp(ip,l)    = NaN;
        lp(ip,l)    = NaN;
        vol(ip,l)   = NaN;
        Levels{l}.NP = L.NP -1;
        
      end
    end  %np
  end  % levels
  
  % look for gaps in the arrays and eliminate the gaps if they are found
  % a gap is a NaN followed by a non-zero number
  for l=1:Limits.maxLevels   
    L = Levels{l};

    NP = length(xp(:,l)) -1;
    for ip=1:NP

      if ( isnan(xp(ip,l)) && ~isnan(xp(ip +1,l)) && xp(ip+1,l) ~= 0.0)
        fprintf(' A gap has been found at L-%g, ip:%g \n',l,ip);
        xp(ip, l)    = xp(ip+1,l);
        massP(ip, l) = massP(ip+1,l);  
        velP(ip, l)  = velP(ip+1,l);
        Fp(ip, l)    = Fp(ip+1,l);     
        lp(ip, l)    = lp(ip+1,l);     
        vol(ip,l)    = vol(ip+1,l);

        xp(ip+1,l)    = NaN;
        massP(ip+1,l) = NaN;
        velP(ip+1,l)  = NaN;
        Fp(ip+1,l)    = NaN;
        lp(ip+1,l)    = NaN;
        vol(ip+1,l)   = NaN;

      end
    end  % np
  end  % levels
  

  
end


%__________________________________
function plotResults(titleStr,t, tstep, P, nodePos, velG, massG, momG, extForceG,intForceG, Limits, Levels)
  global gf;
  global dumpFrames;
    % plot SimulationState
  % convert multilevel arrays into 1D arrays
  
  
  [xp_1D, velP_1D, massP_1D, stressP_1D, Fp_1D ] = ML_to_1D( Levels, Limits, P.xp, P.velP, P.massP, P.stressP,P.Fp);  
  [nodePos_1D, extForceG_1D, intForceG_1D, velG_1D] = ML_Grid_to_1D( Levels, Limits, nodePos, extForceG, intForceG, velG);
         
  fig1 = sfigure(1);
  set(fig1,'position',[10,10,700,700]);
  levelColors = ['m','g','r','b','k'];
  
  subplot(4,1,1),plot(xp_1D,velP_1D,'rd');
  xlim( [Levels{1}.min Levels{1}.max] )
  ylim([min(min(velP_1D - 1e-3) )  max( max(velP_1D + 1e-3))])
  
  xlabel('Particle Position');
  ylabel('Particle velocity');
  title(titleStr);

  subplot(4,1,2),plot(xp_1D,Fp_1D,'rd');
  xlim( [Levels{1}.min Levels{1}.max] )
  ylabel('Fp');

  subplot(4,1,3),plot(xp_1D,stressP_1D,'rd');
  xlim( [Levels{1}.min Levels{1}.max] )
  ylabel('Particle stress');
  
  subplot(4,1,4),plot(xp_1D,massP_1D,'rd');
  xlim( [Levels{1}.min Levels{1}.max] )
  ylabel('Particle mass');
  
  drawNodes(fig1,nodePos,Levels,Limits)

  fig2 = sfigure(2);
  set(fig2,'position',[10,1000,700,700]);
  subplot(3,1,1),plot(nodePos_1D, velG_1D,'bx');
  xlabel('NodePos');
  ylabel('grid Vel');
  title(titleStr);
  xlim( [Levels{1}.min Levels{1}.max] )
  
  
  subplot(3,1,2),plot(nodePos_1D, extForceG_1D,'rd');
  xlim( [Levels{1}.min Levels{1}.max] )
  ylabel('extForceG');
  
  subplot(3,1,3),plot(nodePos_1D, intForceG_1D,'rd');
  xlim( [Levels{1}.min Levels{1}.max] )
  ylabel('intForceG');
  
  if(dumpFrames)
    f_name = sprintf('%g.1.ppm',tstep-1);
    F = getframe(fig1);
    [X,map] = frame2im(F);
    imwrite(X,f_name);
  
    f_name = sprintf('%g.2.ppm',tstep-1);  
    F = getframe(fig2);                     
    [X,map] = frame2im(F);                 
    imwrite(X,f_name);   
  end                  

if(0)
  grad_velG = diff(velG);
  grad_velG(1) = 0.0;
  grad_velG(length(velG)) = 0;
  
  for n=2:length(velG)
    grad_velG(n) = grad_velG(n)/(nodePos(n) - nodePos(n-1) );
  end
  
  length(nodePos)
  length(grad_velG)
  
  subplot(2,1,2),plot(nodePos, grad_velG,'bx');
  ylabel('grad velG');
  xlim( [Levels{1}.min Levels{1}.max] )
end  
  %subplot(6,1,5),plot(nodePos, massG,'bx');
  %ylabel('gridMass');
  %axis([0 50 0 1.1] )
  
  %momG = velG .* massG;
  %subplot(6,1,6),plot(nodePos, momG,'bx');
  %ylabel('gridMom');
  %axis([0 50 0 101] )

  %f_name = sprintf('%g.ppm',tstep-1);
  %F = getframe(gcf);
  %[X,map] = frame2im(F);
  %imwrite(X,f_name)
  %input('hit return');
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
function drawNodes(figure1,nodePos,Levels,Limits)
  % draw vertical lines at each node 
  colors = ['m','g','r','b','k'];
  for l=1:Limits.maxLevels
    L = Levels{l};
    
    % first draw all the lines including the extra cells
    NN = L.NN;
    for n=1:NN            
      x = nodePos(n,l);
      [pt1,pt2] = dsxy2figxy([x,x],ylim);
      annotation(figure1,'line',pt1,pt2,'Color',colors(l));
    end
    
    % color the extra cell nodes
    ECN = length(L.EC_nodes);
    for c=1:ECN   
      n = L.EC_nodes(c);         
      x = nodePos(n,l);
      [pt1,pt2] = dsxy2figxy([x,x],ylim);
      annotation(figure1,'line',pt1,pt2,'Color','b');
    end
    
  end
end

%__________________________________
function [accl_extForceP, delta, bodyForce] = ExternalForceAccl(problem_type, delta_0, bodyForce, Material, xp, xp_initial, t, tstep, NP, L1_dx, bar_length)

  density    = Material{1}.density;
  E          = Material{1}.E;
  speedSound = Material{1}.speedSound;
  
  delta = -9;
  
  nn = length(xp);
  accl_extForceP = zeros(nn,1);  % you must declare arrays that are not passed in.
                                 % This array must be full of zeros
  
  if (strcmp(problem_type, 'compaction'))
    if(bodyForce > -200)
      bodyForce = -t * 1000;
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


 %__________________________________
 % Convert multi-level particle arrays into a 1D array
 % This is mainly used for plotting 
function [xp_1D,varargout] = ML_to_1D( Levels, Limits, xp, varargin)
  
  for i = 1:length(varargin)    % loop over each input argument
    varP = varargin{i};
    
    c = 0;
    for l=1:Limits.maxLevels
      L = Levels{l};
      for ip =1:L.NP
        c = c+1;
        xp_1D(c)   = xp(ip,l);
        varP_1D(c) = varP(ip,l);
      end
    end
    % sort xp into ascending order and varP_1D
    [xp_1D, k] = sort(xp_1D);
    varP_1D = varP_1D(k);
    
    varargout{i} = varP_1D;
  end  % loop over input variables
end  


 %__________________________________
 % Convert multi-level Grid arrays into 1D array
 % This is mainly used for plotting 
function [nodePos_1D, varargout ] = ML_Grid_to_1D( Levels, Limits, nodePos, varargin)
  global gf;
  
  for i = 1:length(varargin)    % loop over each input argument
    varG = varargin{i};
    
    c = 0;
    for l=1:Limits.maxLevels
      L = Levels{l};
      for ig =1:L.NN
        test = gf.hasFinerCell(nodePos(ig,l),l,Levels,Limits, 'withExtraCells');
        if( test == 0)
          c = c+1;
          varG_1D(c) = varG(ig,l);
          nodePos_1D(c)  = nodePos(ig,l);
        end
      end
    end
    
     % sort nodePos into ascending order and varG_1D
    [nodePos_1D, k] = sort(nodePos_1D);
    varG_1D = varG_1D(k);

    varargout{i} = varG_1D;
  end  % loop over input variables
end   
