% Reference:  Jin Ma, Honbing Lu,and Ranga Komanduri, "Structured Mesh
% Refinement in Generalized Interpolation Material Point (GIMP) Method
% for Simulation of Dynamic Problems," CMES, vol. 12, no.3, pp. 213-227, 2006
%______________________________________________________________________
                                                         
   
   

function [L2_norm,maxError,NN,NP]=amrmpm(problem_type,CFL,NCells)
unix('/bin/rm sl_1 sl_2 sl_0')
close all
intwarning on

global d_debugging;
global PPC;
global sf;
global NSFN;               % number of shape function nodes
global dumpFrames;
d_debugging = problem_type;

[mms] = MMS;                      % load mms functions
[sf]  = shapeFunctions;           % load all the shape functions
[IF]  = initializationFunctions;  % load initialization functions

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
  fprintf('usage:  amrmpm(problem type, cfl, R1_dx)\n');
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
  NSFN    = 3;               % Number of shape function nodes linear:2, GIMP:3
else
  NSFN    = 2;        
end

t_initial  = 0.0;              % physical time
tstep     = 0;                % timestep
bodyForce = 0;

BigNum     = int32(1e5);
d_smallNum = double(1e-16);

bar_min     = 0 ;
bar_max     = 50 ;

%bar_min     = 0.75
%bar_max     = 0.83

bar_length  = bar_max - bar_min;

domain       = 52.0;
area         = 1.;
plotSwitch   = 1;
plotInterval = 200;
dumpFrames   = 1;
writeData    = 0;
max_tstep    = 1000;

% HARDWIRED FOR TESTING
%NN          = 16;
R1_dx       =domain/(NCells)

if (mod(domain,R1_dx) ~= 0)
  fprintf('ERROR, the dx in Region 1 does not divide into the domain evenly\n');
  fprintf('usage:  amrmpm(problem type, cfl, R1_dx)\n');
  return;
end

[Regions, nRegions,NN, dx_min] = IF.initialize_Regions(domain,PPC,R1_dx,interpolation, d_smallNum);


% define the boundary condition nodes
BCNodeL(1)  = 1;
BCNodeR(1)  = NN;

if(strcmp(interpolation,'GIMP'))
  BCNodeL(1) = 1;
  BCNodeL(2) = 2;
  BCNodeR(1) = NN-1;
  BCNodeR(2) = NN;
end

%__________________________________
% compute the zone of influence
% compute the positions of the nodes
[nodePos]  = IF.initialize_NodePos(NN, R1_dx, Regions, nRegions, interpolation);
[Lx]       = IF.initialize_Lx(NN, nodePos);

% output the regions and the Lx
nn = 1;
for r=1:nRegions
  R = Regions{r};
  fprintf('-------------------------Region %g\n',r);
  for n=1:R.NN
    fprintf( 'Node:  %g, nodePos: %6.5f, \t Lx(1): %6.5f Lx(2): %6.5f\n',nn, nodePos(nn),Lx(nn,1), Lx(nn,2));
    nn = nn + 1;
  end
end

[xp, NP] = IF.initialize_xp(NN, nodePos, interpolation, PPC, bar_min, bar_max);

%__________________________________
% pre-allocate variables for speed
vol       = zeros(NP,1);
lp        = zeros(NP,1);
massP     = zeros(NP,1);
velP      = zeros(NP,1);
dp        = zeros(NP,1);
stressP   = zeros(NP,1);
Fp        = zeros(NP,1);
dF        = zeros(NP,1);
accl_extForceP = zeros(NP,1);

nodes     = zeros(1,NSFN);
Gs        = zeros(1,NSFN);
Ss        = zeros(1,NSFN);

massG     = zeros(NN,1);
velG      = zeros(NN,1);
vel_nobc_G= zeros(NN,1);
accl_G    = zeros(NN,1);
extForceG = zeros(NN,1);
intForceG = zeros(NN,1);


KE        = zeros(BigNum,1);
SE        = zeros(BigNum,1);
TE        = zeros(BigNum,1);
totalMom  = zeros(BigNum,1);
Exact_tip = zeros(BigNum,1);
DX_tip    = zeros(BigNum,1);
TIME      = zeros(BigNum,1);
totalEng_err   = zeros(BigNum,1);
totalMom_err   = zeros(BigNum,1);
tipDeflect_err = zeros(BigNum,1);


%__________________________________
% initialize other particle variables
for ip=1:NP
  [volP_0, lp_0] = sf.positionToVolP(xp(ip), nRegions, Regions);
  
  vol(ip)   = volP_0;
  massP(ip) = volP_0*density;
  lp(ip)    = lp_0;
  Fp(ip)    = 1.;                     % total deformation
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
  delta_0         = 0;
  velG_BCValueL   = initVelocity;
  velG_BCValueR   = initVelocity;
  for ip=1:NP
    velP(ip)    = initVelocity;
    xp_initial(ip) = xp(ip);
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
  
  xp_initial = zeros(NP,1);
  xp_initial = xp;
  
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
  
  xp_initial = zeros(NP,1);
  xp_initial = xp;
  [Fp]      = mms.deformationGradient(xp_initial, t_initial, NP,speedSound, bar_length);
  [dp]      = mms.displacement(       xp_initial, t_initial, NP, speedSound, bar_length);
  [velP]    = mms.velocity(           xp_initial, t_initial, NP, speedSound, bar_length);
  [stressP] = computeStress(E,Fp,NP);
  
  lp  = lp .* Fp;
  vol = vol .* Fp;
  xp  =  xp_initial + dp;
end

%__________________________________
titleStr(2) = {sprintf('Computational Domain 0,%g, MPM bar %g,%g',domain,bar_min, bar_max)};
titleStr(3) = {sprintf('%s, PPC: %g',interpolation, PPC)};
%titleStr(4)={'Variable Resolution, Center Region refinement ratio: 2'}
titleStr(4) ={sprintf('Composite Grid, #cells %g', NN)};


%plot initial conditions
if(plotSwitch == 1)
  plotResults(titleStr, t_initial, tstep, xp, dp, massP, Fp, velP, stressP, nodePos, velG, massG, extForceG,intForceG)
end
fprintf('t_final: %g, interpolator: %s, NN: %g, NP: %g dx_min: %g \n',t_final,interpolation, NN,NP,dx_min);
%input('hit return')

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

  fid0 = fopen('sl_0','a');
  fid1 = fopen('sl_1','a');
  fid2 = fopen('sl_2','a');
  fprintf(fid0,'__________________________________\n\n');
  fprintf(fid1,'__________________________________\n\n');
  fprintf(fid2,'__________________________________\n\n');
  fclose(fid0);
  fclose(fid1);
  fclose(fid2);


  % compute the timestep
  dt = double(BigNum);
  for ip=1:NP
    dt = min(dt, CFL*dx_min/(speedSound + abs(velP(ip) ) ) );
  end

  tstep = tstep + 1;
  t = t + dt;
  if (mod(tstep,20) == 0)
    fprintf('timestep %g, dt = %g, time %g \n',tstep, dt, t)
  end
  
  % initialize arrays to be zero
  for ig=1:NN
    massG(ig)     =1.e-100;
    velG(ig)      =0.;
    vel_nobc_G(ig)=0.;
    accl_G(ig)    =0.;
    extForceG(ig) =0.;
    intForceG(ig) =0.;
  end
  
  % compute the problem specific external force acceleration.
  [accl_extForceP, delta,bodyForce] = ExternalForceAccl(problem_type, delta_0, bodyForce, Material, xp, xp_initial, t, tstep, NP, R1_dx, bar_length);
    
  %__________________________________
  % project particle data to grid  
  for ip=1:NP
  
    [nodes,Ss]=sf.findNodesAndWeights_linear(xp(ip), lp(ip), nRegions, Regions, nodePos, Lx);
    for ig=1:NSFN
      massG(nodes(ig))     = massG(nodes(ig))     + massP(ip) * Ss(ig);
      velG(nodes(ig))      = velG(nodes(ig))      + massP(ip) * velP(ip) * Ss(ig);
      extForceG(nodes(ig)) = extForceG(nodes(ig)) + massP(ip) * accl_extForceP(ip) * Ss(ig); 
      
      % debugging__________________________________
      n = nodePos(nodes(ig));
      %if( ( (n >= 17.333 && n <= 17.34) || (n >= 34.6 && n <= 34.7) ) )
      if( (n >= 34.6 && n <= 34.7))
        fid = fopen('sl_0','a');
        fprintf(fid, 'ip: %g xp: %g, nodes: %g, node_pos: %g massG: %g, massP: %g, Ss: %g,  prod: %g \n', ip, xp(ip), nodes(ig), nodePos(nodes(ig)), massG(nodes(ig)), massP(ip), Ss(ig), massP(ip) * Ss(ig) );
        fprintf(fid, '\t velG:      %g,  velP:       %g,  prod: %g \n', velG(nodes(ig)), velP(ip), (massP(ip) * velP(ip) * Ss(ig) ) );
        fprintf(fid, '\t extForceG: %g,  accl_extForceP:  %g,  prod: %g \n', extForceG(nodes(ig)), accl_extForceP(ip), accl_extForceP(ip) * Ss(ig) );
        fclose(fid);
      end 
      % debugging__________________________________

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
  for ip=1:NP
    [nodes,Gs,dx]=sf.findNodesAndWeightGradients_linear(xp(ip), lp(ip), nRegions, Regions, nodePos,Lx);
    for ig=1:NSFN
      intForceG(nodes(ig)) = intForceG(nodes(ig)) - Gs(ig) * stressP(ip) * vol(ip);
      
      n = nodePos(nodes(ig));
      %if( ( (n >= 17.333 && n <= 17.34) || (n >= 34.6 && n <= 34.7) ) )
      if( (n >= 34.6 && n <= 34.7))
         source = Gs(ig) * stressP(ip) * vol(ip);
         fid = fopen('sl_1','a');
         fprintf( fid,'nodePos:%g ip: %g internalForceG: %g source: %g, stressP: %g vol: %g Gs: %g\n',n, ip,intForceG(nodes(ig)), source, stressP(ip), vol(ip), Gs(ig)   );
         fclose(fid);
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
      exact_pos = (xp_initial(ip) + t * initVelocity);
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
    
    fig3 = sfigure(3);
    set(fig3,'position',[1000,100,700,700]);

    plot(xp,stressP,'rd', xp, stressExact, 'b');
    axis([0 50 -10000 0])

    title(titleStr)
    legend('Simulation','Exact')
    xlabel('Position');
    ylabel('Particle stress');

    if(dumpFrames)
      f_name = sprintf('%g.3.ppm',tstep-1);
      F = getframe(gcf);
      [X,map] = frame2im(F);
      imwrite(X,f_name);
    end
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
    plotResults(titleStr, t, tstep, xp, dp, massP, Fp, velP, stressP, nodePos, velG, massG, extForceG,intForceG)
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
      
      %if(xp(ip) >= 17.333 && xp(ip) <= 18.45) 
      if(1)
      if(xp(ip) >=  34.156 && xp(ip) <= 35.17) 
        fid = fopen('sl_2','a');
        fprintf(fid,'nodePos:%g xp:%g, gUp:%16.15E, velG:%16.15E Gs:%16.15E \n',nodePos(nodes(ig)),xp(ip), gUp, velG(nodes(ig)), Gs(ig));
        fclose(fid);
      end
      end
      
    end

    dF(ip)      = 1. + gUp * dt;
    Fp(ip)      = dF(ip) * Fp(ip);
    vol(ip)     = volP_0 * Fp(ip);
    lp(ip)      = lp_0 * Fp(ip);
    
    %if(xp(ip) >= 17.333 && xp(ip) <= 18.45) 
    if(0)
    if(xp(ip) >=  34.156 && xp(ip) <= 35.17)
      fid = fopen('sl_2','a');
      fprintf(fid,'xp:%g, gUp:%16.15E, Fp:%16.15E \n',xp(ip), gUp, Fp(ip));
      fclose(fid);
    end
    end 
  end
end


%__________________________________
function [stressP]=computeStress(E,Fp,NP)
                                                                                
  for ip=1:NP
    stressP(ip) = E * (Fp(ip)-1.0);
%    stressP(ip) = (E/2.0) * ( Fp(ip) - 1.0/Fp(ip) );        % hardwired for the mms test  see eq 50
  end
end




%__________________________________
function plotResults(titleStr,t, tstep, xp, dp, massP, Fp, velP, stressP, nodePos, velG, massG, extForceG,intForceG)
  global dumpFrames;
  
  % plot SimulationState
  fig1 = sfigure(1);
  set(fig1,'position',[50,100,700,700]);
  
  subplot(4,1,1),plot(xp,velP,'rd');
  ylim([min(velP - 1e-3) max(velP + 1e-3)])
  xlim([min(nodePos) max(nodePos)]);
  xlabel('Particle Position');
  ylabel('Particle velocity');
  title(titleStr);

  subplot(4,1,2),plot(xp,Fp,'rd');
  xlim([min(nodePos) max(nodePos)]);
  ylabel('Fp');

  subplot(4,1,3),plot(xp,stressP,'rd');
  xlim([min(nodePos) max(nodePos)]);
  ylabel('Particle stress');
  
  subplot(4,1,4),plot(xp,massP,'rd');
  xlim([min(nodePos) max(nodePos)]);
  ylabel('Particle mass');
  
  NN = length(nodePos);
  for n=1:NN           
    x = nodePos(n);
    [pt1,pt2] = dsxy2figxy([x,x],ylim);
    annotation(fig1,'line',pt1,pt2,'Color','r');
  end
  
  %__________________________________
  fig2 = sfigure(2);
  set(fig2,'position',[10,1000,700,700]);
  subplot(3,1,1),plot(nodePos, velG,'bx');
  xlabel('NodePos');
  ylabel('grid Vel');
  title(titleStr);
  xlim([min(nodePos) max(nodePos)]);
  
  subplot(3,1,2),plot(nodePos, extForceG,'rd');
  xlim([min(nodePos) max(nodePos)]);
  ylabel('extForceG');
  
  subplot(3,1,3),plot(nodePos, intForceG,'rd');
  xlim([min(nodePos) max(nodePos)]);
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
function [accl_extForceP, delta, bodyForce] = ExternalForceAccl(problem_type, delta_0, bodyForce, Material, xp, xp_initial, t, tstep, NP, R1_dx, bar_length)

  density    = Material{1}.density;
  E          = Material{1}.E;
  speedSound = Material{1}.speedSound;
  
  delta = -9;
  
  accl_extForceP = zeros(length(xp),1);
  
  if (strcmp(problem_type, 'compaction'))
    if(bodyForce > -200)
      bodyForce = -t * 1000;
    end
    
    delta = delta_0 + (density*bodyForce/(2.0 * E) ) * (delta_0 * delta_0);  

    displacement = delta - delta_0;
    W = density * abs(bodyForce) * delta_0; 
    
    if (mod(tstep,100) == 0)
      fprintf('Bodyforce: %g displacement:%g, W: %g\n',bodyForce, displacement/R1_dx, W);                                             
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
