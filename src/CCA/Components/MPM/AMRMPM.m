%Reference:  Jin Ma, Honbing Lu,and Ranga Komanduri, "Structured Mesh
% Refinement in Generalized Interpolation Material Point (GIMP) Method
% for Simulation of Dynamic Problems," CMES, vol. 12, no.3, pp. 213-227, 2006

%______________________________________________________________________



function KK=amrmpm(problem_type,CFL,NN)
close all

global d_debugging;
global PPC;
d_debugging = problem_type

%  valid options:
%  problem type:  impulsiveBar, oscillator, collidingBars
%  CFL:            0 < cfl < 1, usually 0.2 or so.
%  R1_dx:          cell spacing in region 1.
% One dimensional MPM
% bulletproofing

if (~strcmp(problem_type, 'impulsiveBar')  && ...
    ~strcmp(problem_type, 'oscillator')    && ...
    ~strcmp(problem_type, 'collidingBars') && ...
    ~strcmp(problem_type, 'compaction')    && ...
    ~strcmp(problem_type, 'advectBlock') )
  fprintf('ERROR, the problem type is invalid\n');
  fprintf('     The valid types are impulsiveBar, oscillator, collidingBars\n');
  fprintf('usage:  amrmpm(problem type, cfl, R1_dx)\n');
  return;
end
%__________________________________
% hard coded domain
PPC     = 2;
E       = 1e6;
density = 1.;
c       = sqrt(E/density);


BigNum     = int32(1e5);
d_smallNum = double (1e-16);

bar_min     = 0.0;
bar_max     = 50;

bar_length     = bar_max - bar_min;

domain     = 52;
area       = 1.;
plotSwitch = 0;
max_tstep  = BigNum

% HARDWIRED FOR TESTING
%NN          = 16;
R1_dx       =domain/(NN-1)

if (mod(domain,R1_dx) ~= 0)
  fprintf('ERROR, the dx in Region 1 does not divide into the domain evenly\n');
  fprintf('usage:  amrmpm(problem type, cfl, R1_dx)\n');
  return;
end

%__________________________________
% region structure 



%____________
% single level
numRegions    = int32(2);               % partition the domain into numRegions
Regions       = cell(numRegions,1);    % array that holds the individual region information
R.min         = 0;                     % location of left point
R.max         = domain/2;                   % location of right point
R.refineRatio = 1;
R.dx          = R1_dx;
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx +1 );
Regions{1}    = R;

R.min         = domain/2;                       
R.max         = domain;
R.refineRatio = 1;
R.dx          = R1_dx/R.refineRatio;
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx );
Regions{2}    = R;

%____________
% 2 level
if(0)
numRegions    = int32(3);               % partition the domain into numRegions
Regions       = cell(numRegions,1);    % array that holds the individual region information

R.min         = 0;                     % location of left point
R.max         = 0.32;                   % location of right point
R.refineRatio = 1;
R.dx          = R1_dx;
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx +1 );
Regions{1}    = R;

R.min         = 0.32;                       
R.max         = 0.64;
R.refineRatio = 2;
R.dx          = R1_dx/R.refineRatio;
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx );
Regions{2}    = R;

R.min         = 0.64;                       
R.max         = domain;
R.refineRatio = 1;
R.dx          = R1_dx/R.refineRatio; 
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx);       
Regions{3}    = R;

end

%____________
% 3 levels
if(0)

numRegions    = int32(5);               % partition the domain into numRegions
Regions       = cell(numRegions,1);    % array that holds the individual region information

R.min         = 0;                     % location of left point
R.max         = 0.32;                   % location of right point
R.refineRatio = 1;
R.dx          = R1_dx;
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx +1 );
Regions{1}    = R;

R.min         = 0.32;                       
R.max         = 0.4;
R.refineRatio = 4;
R.dx          = R1_dx/double(R.refineRatio);
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx );
Regions{2}    = R;

R.min         = 0.4;                       
R.max         = 0.56;
R.refineRatio = 16;
R.dx          = R1_dx/double(R.refineRatio); 
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx);       
Regions{3}    = R;

R.min         = 0.56;                       
R.max         = 0.64;
R.refineRatio = 4;
R.dx          = R1_dx/double(R.refineRatio);
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx );
Regions{4}    = R;

R.min         = 0.64;                       
R.max         = domain;
R.refineRatio = 1;
R.dx          = R1_dx/R.refineRatio;
R.volP        = R.dx/PPC;
R.NN          = int32( (R.max - R.min)/R.dx );
Regions{5}    = R;

end

NN = int32(0);

% bulletproofing:
for r=1:numRegions
  R = Regions{r};
  d = (R.max - R.min) + 100* d_smallNum;
  
  if( mod( d, R.dx ) > 1.0e-10 )
    fprintf('ERROR, the dx: %g in Region %g does not divide into the domain (R.max:%g R.min:%g) evenly\n', R.dx,r,R.max,R.min);
    return;
  end
end


%  find the number of nodes (NN) and the minimum dx
dx_min = double(BigNum);
for r=1:numRegions
  R = Regions{r};
  NN = NN + R.NN;
  dx_min = min(dx_min,R.dx);
  fprintf( 'region %g, min: %g, \t max: %g \t refineRatio: %g dx: %g, NN: %g\n',r, R.min, R.max, R.refineRatio, R.dx, R.NN)
end

%__________________________________
% compute the zone of influence
% compute the positions of the nodes
Lx      = zeros(NN,2);
nodePos = zeros(NN,1);      % node Position
nodeNum = int32(1);

nodePos(1) = 0.0;

for r=1:numRegions
  R = Regions{r};
  
  % determine dx_Right and dx_Left of this region
  if(r == 1)               % leftmost region
    dx_L = 0;
    dx_R = Regions{r+1}.dx;
  elseif(r == numRegions)  % rightmost region
    dx_L = Regions{r-1}.dx;
    dx_R = 0;
  else                     % all other regions
    dx_L = Regions{r-1}.dx;
    dx_R = Regions{r+1}.dx;
  end
  
  % loop over all nodes and set Lx minus/plus
  for  n=1:R.NN
    if(n == 1)
      Lx(nodeNum,1) = dx_L;
      Lx(nodeNum,2) = R.dx;
    elseif(n == R.NN)
      Lx(nodeNum,1) = R.dx;
      Lx(nodeNum,2) = dx_R;
    else
      Lx(nodeNum,1) = R.dx;
      Lx(nodeNum,2) = R.dx;
    end
    
    if(nodeNum > 1)
      nodePos(nodeNum) = nodePos(nodeNum-1) + R.dx;
    end
    nodeNum = nodeNum + 1;
  end
end


% output the regions and the Lx
nn = 1;
for r=1:numRegions
  R = Regions{r};
  fprintf('-------------------------Region %g\n',r);
  for n=1:R.NN
    fprintf( 'Node:  %g, nodePos: %6.5f, \t Lx(1): %6.5f Lx(2): %6.5f\n',nn, nodePos(nn),Lx(nn,1), Lx(nn,2));
    nn = nn + 1;
  end
end

%__________________________________
% create particles
ip=1;
xp(1)=bar_min + R1_dx/(2.0 * PPC);

for r=1:numRegions
  R = Regions{r};
  
  dx_P = R.dx/PPC;                             % particle dx
  
  if ~strcmp(problem_type, 'collidingBars')    % Everything except the collingBar
    while (xp(ip) + dx_P > R.min )   && ...
          (xp(ip) + dx_P < R.max )   && ...
          (xp(ip) + dx_P >= bar_min) && ...
          (xp(ip) + dx_P <= bar_max)
          
      ip = ip+1;
      xp(ip)=xp(ip-1) + dx_P;
    end
  end
  
 % This needs to be fixed!!!                    % Colliding Bar
  if strcmp(problem_type, 'collidingBars')
    %left bar
    while xp(ip) + dx_P < (bar_length/2. - R.dx)
      ip=ip+1;
      xp(ip)=xp(ip-1) + dx_P;
    end

    ip=ip+1;
    xp(ip)=domain - R.dx/(2.*PPC);

    % right bar
    while xp(ip) - dx_P > (bar_length/2. + R.dx)
      ip=ip+1;
      xp(ip)=xp(ip-1) - dx_P;
    end
  end
end  % region

NP=ip;  % number of particles


%__________________________________
% pre-allocate variables for speed
vol       = zeros(NP,1);
massP     = zeros(NP,1);
velP      = zeros(NP,1);
dp        = zeros(NP,1);
stressP   = zeros(NP,1);
extForceP = zeros(NP,1);
Fp        = zeros(NP,1);
nodes     = zeros(1,2);
Gs        = zeros(1,2);
Ss        = zeros(1,2);     

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
  [volP]    = positionToVolP(xp(ip), numRegions, Regions);
  vol(ip)   = volP;
  massP(ip) = volP*density;
  Fp(ip)    = 1.;                     % total deformation
end

%______________________________________________________________________
% problem specific 

BCNode(1)  = 1;
BCNode(2)  = NN;

if strcmp(problem_type, 'impulsiveBar')
  period          = sqrt(16.*bar_length*bar_length*density/E);
  tfinal          = period;
  TipForce        = 10.;
  D               = TipForce*bar_length/(area*E);
  M               = 4.*D/period;
  extForceP(NP)   = TipForce;
  velG_BCValue(1) = 0.;
  velG_BCValue(2) = 1.;
end

if strcmp(problem_type, 'oscillator')
  Mass            = 10000.;
  period          = 2.*3.14159/sqrt(E/Mass);
  tfinal          = period;
  v0              = 0.5;
  Amp             = v0/(2.*3.14159/period);
  massP(NP)       = Mass;                    % last particle masss
  velP(NP)        = v0;                      % last particle velocity
  numBCs          = 1;
  velG_BCValue(1) = 0.;
  velG_BCValue(2) = 1.;
end

if strcmp(problem_type, 'collidingBars')
  period           = 4.0*dx/100.;
  tfinal           = period;
  numBCs           = 0;
  velG_BCValue(1)  = 0.;
  velG_BCValue(2)  = 1.;
  for ip=1:NP
    velP(ip) =100.0;
    
    if xp(ip) > .5*bar_length
      velP(ip) = -100.0;
    end
  end
end

if strcmp(problem_type, 'advectBlock')
  initVelocity    = 100;
  tfinal          = 0.01;
  numBCs          = 1;
  velG_BCValue(1) = initVelocity;
  velG_BCValue(2) = initVelocity;
  for ip=1:NP
    velP(ip)    = initVelocity;
    initPos(ip) = xp(ip);
  end
end

if strcmp(problem_type, 'compaction')
  initVelocity    = 0;
  waveTansitTime  = bar_length/c
  tfinal          = waveTansitTime * 45;
  numBCs          = 1;
  delta_0         = 50
  velG_BCValue(1) = initVelocity;
  velG_BCValue(2) = initVelocity;
end


%plot initial conditions
%plotResults(t,xp,dp,velP)

fprintf('tfinal: %g, NN: %g, NP: %g dx_min: %g \n',tfinal, NN,NP,dx_min);
input('hit return')

%==========================================================================
% Main timstep loop

t = 0.0;
tstep = 0;
bodyForce = 0;

while t<tfinal && tstep < max_tstep

  % compute the timestep
  dt = double(BigNum);
  for ip=1:NP
    dt = min(dt, CFL*dx_min/(c + abs(velP(ip) ) ) );
  end

  tstep = tstep + 1;
  t = t + dt;
  if (mod(tstep,100) == 0)
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
  
  
  %__________________________________
  % debugging
  if (strcmp(problem_type, 'compaction'))
    if(bodyForce > -200)
      bodyForce = -t * 100;
    end
    
    delta = delta_0 + (density*bodyForce/(2.0 * E) ) * (delta_0 * delta_0);  

    displacement = delta - delta_0;
    W = density * abs(bodyForce) * delta_0; 
    
    if (mod(tstep,100) == 0)
      fprintf('Bodyforce: %g displacement:%g, W: %g\n',bodyForce, displacement/R1_dx, W);                                             
    end
    for ip=1:NP                                                                       
       extForceP(ip) = bodyForce*massP(ip);                                                      
    end                                                                               
  end
  %__________________________________
  
    
  %__________________________________
  % project particle data to grid  
  for ip=1:NP
    [nodes,Ss]=findNodesAndWeights_gimp(xp(ip), numRegions, Regions, nodePos, Lx);
    for ig=1:2
      massG(nodes(ig))     = massG(nodes(ig))     + massP(ip) * Ss(ig);
      velG(nodes(ig))      = velG(nodes(ig))      + massP(ip) * velP(ip) * Ss(ig);
      extForceG(nodes(ig)) = extForceG(nodes(ig)) + extForceP(ip) * Ss(ig); 
      
      % debugging__________________________________
      if(0)
        fprintf( 'ip: %g xp: %g, nodes: %g, node_pos: %g massG: %g, massP: %g, Ss: %g,  prod: %g \n', ip, xp(ip), nodes(ig), nodePos(nodes(ig)), massG(nodes(ig)), massP(ip), Ss(ig), massP(ip) * Ss(ig) );
        fprintf( '\t velG:      %g,  velP:       %g,  prod: %g \n', velG(nodes(ig)), velP(ip), (massP(ip) * velP(ip) * Ss(ig) ) );
        fprintf( '\t extForceG: %g,  extForceP:  %g,  prod: %g \n', extForceG(nodes(ig)), extForceP(ip), extForceP(ip) * Ss(ig) );
      end 
      % debugging__________________________________

    end
  end

  % normalize by the mass
  velG = velG./massG;
  vel_nobc_G = velG;

  % set velocity BC
  for ibc=1:numBCs
    velG(BCNode(ibc)) = velG_BCValue(ibc);
  end

  % debugging__________________________________ 
  if( strcmp(problem_type, 'advectBlock')  )
    for ig=1:NN
      error = velG(ig) * massG(ig) - massG(ig) * initVelocity;
      if(  (abs(error) > 1e-8) )
        fprintf('interpolateParticlesToGrid after BC:  node: %g, nodePos %g, error %g, massG %g \n', ig, nodePos(ig), error, massG(ig) );
      end
    end
  end
  % debugging__________________________________

  %compute particle stress
  [stressP,vol,Fp]=computeStressFromVelocity(xp,dt,velG,E,Fp,NP,numRegions, Regions, nodePos);

  %compute internal force
  for ip=1:NP
    [nodes,Gs,dx]=findNodesAndWeightGradients_gimp(xp(ip),numRegions, Regions, nodePos);
    for ig=1:2
      intForceG(nodes(ig)) = intForceG(nodes(ig)) - Gs(ig) * stressP(ip) * vol(ip);
    end
  end

% debugging__________________________________
  if( strcmp(problem_type, 'advectBlock')  )
    if( abs(intForceG(nodes(ig))) > 1e-8 )
      fprintf('internal Force: \t  node: %g, nodePos %g, intForce %g \n', nodes(ig), nodePos(nodes(ig)), intForceG(nodes(ig)) );
      input('hit return')
    end
  end
% debugging__________________________________

  %compute the acceleration and new velocity on the grid
  accl_G    =(intForceG + extForceG)./massG;
  vel_new_G = velG + accl_G.*dt;
  

  %set velocity BC
  for ibc=1:numBCs
    vel_new_G(BCNode(ibc)) = velG_BCValue(ibc);
  end

  momG = massG .* vel_new_G;

  % compute the acceleration on the grid
  for ig=1:NN
    accl_G(ig)  = (vel_new_G(ig) - vel_nobc_G(ig))/dt;
  end
  
  %__________________________________
  %project changes back to particles
  tmp = zeros(NP,1);
  for ip=1:NP
    [nodes,Ss]=findNodesAndWeights_gimp(xp(ip),numRegions, Regions, nodePos, Lx);
    dvelP = 0.;
    dxp   = 0.;
    
    for ig=1:2
      dvelP = dvelP + accl_G(nodes(ig))    * dt * Ss(ig);
      dxp   = dxp   + vel_new_G(nodes(ig)) * dt * Ss(ig);
      
% debugging__________________________________
  if( strcmp(problem_type, 'advectBlock')  )
    error = massG(nodes(ig)) * vel_new_G(nodes(ig)) - massG(nodes(ig)) * initVelocity;
    if(  (abs(error) > 1e-8) && (massG(nodes(ig)) > 0) )
      fprintf('project changes: \t  node: %g, nodePos %g, error %g, massG %g \n', nodes(ig), nodePos(nodes(ig)), error, massG(nodes(ig) ));
      fprintf(' \t\t\t vel_new_G: %g  Ss(ig): %16.15f\n',vel_new_G(nodes(ig)), Ss(ig) );
    end
  end
% debugging__________________________________
    
    end
    
    velP(ip) = velP(ip) + dvelP;
    xp(ip)   = xp(ip) + dxp;
    dp(ip)   = dp(ip) + dxp;
    tmp(ip)  = tmp(ip) + dxp;
  end
  
  %fprintf('sum(tmp): %9.8f \n',sum(tmp));
  

  
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
      %fprintf('xp: %f  exact: %f error %f \n',xp(ip), exact_pos, xp(ip) - exact_pos)
    end
    fprintf('sum position error %16.15f \n',sum(pos_error))
  end
  
  if strcmp(problem_type, 'compaction')
    term1 = (2.0 * density * bodyForce)/E;
    for ip=1:NP
      term2 = term1 * (delta - xp(ip));
      stressExact(ip) = E *  ( sqrt( term2 + 1.0 ) - 1.0);
    end
    
    if (mod(tstep,200) == 0) 
  
      set(gcf,'position',[50,100,700,500]);
      figure(1)
      plot(xp,stressP,'rd', xp, stressExact, 'b');
      axis([0 50 -10000 0])
      title('Quasi-Static Compaction Problem, Single Level \Delta{x} = 0.5, PPC: 1, Cells: 100')
      legend('Simulation','Exact')
      xlabel('Position');
      ylabel('Particle stress');

      f_name = sprintf('%g.ppm',tstep-1);
      F = getframe(gcf);
      [X,map] = frame2im(F);
      imwrite(X,f_name);
      
      d = abs(stressP - stressExact);
      L2_norm = sqrt( sum(d.^2)/length(stressP) )
      fid = fopen('L2norm.dat', 'a');
      fprintf(fid,'%g %g\n',t, L2_norm);
      fclose(fid);
      
    end
  end
  

  TIME(tstep)=t;
  
  %__________________________________
  % plot intantaneous solution
  if (mod(tstep,10) == 0) && (plotSwitch == 1)
    plotResults(t, xp, dp, massP, velP, stressP, nodePos, velG, massG, momG)
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
      t = tfinal;
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
fname1 = sprintf('particle_NN_%g_PPC_%g.dat',NN, PPC);
fid = fopen(fname1, 'w');
fprintf(fid,'#%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
fprintf(fid,'#p, xp, massP, velP, stressP, extForceP, Fp\n');
for ip=1:NP
  fprintf(fid,'%g, %16.15f, %16.15f, %16.15f, %16.15f, %16.15f, %16.15f\n',ip, xp(ip),massP(ip), velP(ip), stressP(ip), extForceP(ip), Fp(ip));
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


%______________________________________________________________________
% functions
function[node, dx]=positionToNode(xp, numRegions, Regions)
 
  n_offset = 0;
  region1_offset = 1;                       % only needed for the first region
  dx = double(0);
  
  for r=1:numRegions
    R = Regions{r};
    
    if ((xp >= R.min) && (xp < R.max))
      n    = floor((xp - R.min)/R.dx);      % # of nodes from the start of the current region
      node = n + n_offset + region1_offset; % add an offset to the local node number
      dx   = R.dx;
      %fprintf( 'region: %g, n: %g, node:%g, xp: %g dx: %g R.min: %g, R.max: %g \n',r, n, node, xp, dx, R.min, R.max);
      return;
    end
    region1_offset = 0;                     % set to 0 after the first region

    n_offset = (n_offset) + R.NN;           % increment the offset
  end
end
%__________________________________
%
function[volP]=positionToVolP(xp, numRegions, Regions)
  volP = -9.0;
 
  for r=1:numRegions
    R = Regions{r};
    if ( (xp >= R.min) && (xp < R.max) )
      volP = R.dx;
    end
  end
end



%__________________________________
function [nodes,Ss]=findNodesAndWeights(xp, numRegions, Regions, nodePos, Lx)
 
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 12 of the reference

  [node, dx]=positionToNode(xp, numRegions, Regions);  

  nodes(1)= node;
  nodes(2)= node+1;

  dnode = double(node);
  
  Lx_minus = Lx(node,1);
  Lx_plus  = Lx(node,2);
  delX = xp - nodePos(node);
    
  if (delX <= -Lx_minus)
    Ss(1) = 0;
    Ss(2) = 1;
  elseif(-Lx_minus <= delX && delX<= 0.0)
    Ss(1) = 1.0 + delX/Lx_minus;
    Ss(2) = 1.0 - Ss(1);
  elseif(  0 <= delX && delX<= Lx_plus)
    Ss(1) = 1.0 - delX/Lx_plus;
    Ss(2) = 1.0 - Ss(1);
  elseif( Lx_plus <= delX )
    Ss(1) = 0;
    Ss(2) = 1;
  end
  
  if(0)
    node = xp/dx;
    node=floor(node)+1;

    nodes(1)= node;
    nodes(2)=nodes(1)+1;

    dnode=double(node);

    locx=(xp-dx*(dnode-1))/dx;
    Ss_old(1)=1-locx;
    Ss_old(2)=locx;

    if ( ( Ss(1) ~= Ss_old(1)) || ( Ss(1) ~= Ss_old(1)) )
      fprintf('SS(1): %g, Ss_old: %g     Ss(2):%g   Ss_old(2):%g \n',Ss(1), Ss_old(1), Ss(2), Ss_old(2));
    end
  end
end


%__________________________________
function [nodes,Ss]=findNodesAndWeights_gimp(xp, numRegions, Regions, nodePos, Lx)
  global PPC;
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 7.16 of the MPM documentation 

  [node, dx]=positionToNode(xp, numRegions, Regions);  

  nodes(1)= node;
  nodes(2)= node+1;
  
  L = dx;
  lp= dx/(2 * PPC);          % This assumes that lp = lp_initial.
  
  for ig=1:2
    Ss(ig) = -9;
    delX = xp - nodePos(nodes(ig));

    if ( ((-L-lp) < delX) && (delX <= -L+lp) )
      
      Ss(ig) = ( ( L + lp + delX)^2 )/ (4.0*L*lp);
      
    elseif( ((-L+lp) < delX) && (delX <= -lp) )
      
      Ss(ig) = 1 + delX/L;
      
    elseif( (-lp < delX) && (delX <= lp) )
      
      numerator = delX^2 + lp^2;
      Ss(ig) =1.0 - (numerator/(2.0*L*lp));  
    
    elseif( (lp < delX) && (delX <= L-lp) )
      
      Ss(ig) = 1 - delX/L;
            
    elseif( (L-lp < delX) && (delX <= L+lp) )
    
      Ss(ig) = ( ( L + lp - delX)^2 )/ (4.0*L*lp);
    
    else
      SS(ig) = 0;
    end
  end
end

%__________________________________
function [nodes,Ss]=findNodesAndWeights_gimp2(xp, numRegions, Regions, nodePos, Lx)
  global PPC;
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 15 of the reference

  [node, dx]=positionToNode(xp, numRegions, Regions);  

  nodes(1)= node;
  nodes(2)= node+1;
  
  Lx_minus = Lx(node,1);
  Lx_plus  = Lx(node,2);
  lp       = dx/(2 * PPC);          % This assumes that lp = lp_initial.
  
  delX = xp - nodePos(node);
  A = delX - lp
  B = delX + lp
  a = max( A, -Lx_minus)
  b = min( B,  Lx_plus)
  
    
  if (B <= -Lx_minus || A >= Lx_plus)
    Ss(1) = 0;
    Ss(2) = 1;
  elseif( b <= 0 )
    t1 = b - a;
    t2 = (b*b - a*a)/(2.0*Lx_minus);
    
    Ss(1) = (t1 + t2)/(2.0*lp);
    Ss(2) = 1.0 - Ss(1);
  elseif( a >= 0 )
    t1 = b - a;
    t2 = (b*b - a*a)/(2.0*Lx_plus);
    
    Ss(1) = (t1 - t2)/(2.0*lp);
    Ss(2) = 1.0 - Ss(1);
  else
    t1 = b - a;
    t2 = (a*a)/(2.0*Lx_minus);
    t3 = (b*b)/(2.0*Lx_plus);
    
    Ss(1) = (t1 - t2 - t3)/(2*lp);
    Ss(2) = 1.0 - Ss(1);
  end
  
end

%__________________________________
function [nodes,Gs, dx]=findNodesAndWeightGradients(xp, numRegions, Regions, nodePos);
 
  % find the nodes that surround the given location and
  % the values of the gradients of the shape functions.
  % Assume the grid starts at x=0.

  [node, dx]=positionToNode(xp,numRegions, Regions);

  nodes(1) = node;
  nodes(2) = nodes(1)+1;

  Gs(1) = -1/dx;
  Gs(2) = 1/dx;
end

%__________________________________
function [nodes,Gs, dx]=findNodesAndWeightGradients_gimp(xp, numRegions, Regions, nodePos);
  global PPC;
  % find the nodes that surround the given location and
  % the values of the gradients of the shape functions.
  % Assume the grid starts at x=0.

  [node, dx]=positionToNode(xp,numRegions, Regions);
  nodes(1)= node;
  nodes(2)= node+1;
  
  L  = dx;
  lp = dx/(2 * PPC);          % This assumes that lp = lp_initial.
  
  for ig=1:2
    Gs(ig) = -9;
    delX = xp - nodePos(nodes(ig));

    if ( ((-L-lp) < delX) && (delX <= -L+lp) )
      
      Gs(ig) = ( L + lp + delX )/ (2.0*L*lp);
      
    elseif( ((-L+lp) < delX) && (delX <= -lp) )
      
      Gs(ig) = 1/L;
      
    elseif( (-lp < delX) && (delX <= lp) )
      
      Gs(ig) =-delX/(L*lp);  
    
    elseif( (lp < delX) && (delX <= L-lp) )
      
      Gs(ig) = -1/L;
            
    elseif( (L-lp < delX) && (delX <= L+lp) )
    
      Gs(ig) = -( L + lp - delX )/ (2.0*L*lp);
    
    else
      Gs(ig) = 0;
    end
  end
end
%__________________________________
function [stressP,vol,Fp]=computeStressFromVelocity(xp,dt,velG,E,Fp,NP, numRegions, Regions, nodePos)
  global d_debugging;
                                                                                
  for ip=1:NP
    [nodes,Gs,dx] = findNodesAndWeightGradients_gimp(xp(ip), numRegions, Regions, nodePos);
    [volP]        = positionToVolP(xp(ip), numRegions, Regions);
    
    gUp=0.0;
    for ig=1:2
      gUp = gUp + velG(nodes(ig)) * Gs(ig);
    end

    dF          =1. + gUp * dt;
    Fp(ip)      = dF * Fp(ip);
    stressP(ip) = E * (Fp(ip)-1.0);
    vol(ip)     = volP * Fp(ip);

    if( strcmp(d_debugging, 'advectBlock') && abs(stressP(ip)) > 1e-8) 
      fprintf('computeStressFromVelocity: nodes_L: %g, nodes_R:%g, gUp: %g, dF: %g, stressP: %g \n',nodes(1),nodes(2), gUp, dF, stressP(ip) );
      fprintf(' Gs_L: %g, Gs_R: %g\n', Gs(1), Gs(2) );
      fprintf(' velG_L: %g, velG_R: %g\n', velG(nodes(1)), velG(nodes(2)) );
      fprintf(' prod_L %g, prod_R: %g \n\n', velG(nodes(1)) * Gs(1), velG(nodes(2)) * Gs(2) );
    end
    
  end
end

%__________________________________
function plotResults(t, xp, dp, massP, velP, stressP, nodePos, velG, massG, momG)
    % plot SimulationState
  set(gcf,'position',[50,100,900,900]);
  figure(1)
  subplot(6,1,1),plot(xp,velP,'rd');
  xlabel('Particle Position');
  ylabel('Particle velocity');
  %axis([0 1 99 101] )

  subplot(6,1,2),plot(xp,massP,'rd');
  %axis([0 1 53 53.5] )
  ylabel('Particle Mass');

  subplot(6,1,3),plot(xp,stressP,'rd');
  
  %axis([0 1 -1 1] )
  ylabel('Particle stress');

  subplot(6,1,4),plot(nodePos, velG,'bx');
  xlabel('NodePos');
  ylabel('grid Vel');
  %axis([0 1 0 110] )

  subplot(6,1,5),plot(nodePos, massG,'bx');
  ylabel('gridMass');
  %axis([0 1 0 70] )

  momG = velG .* massG;
  subplot(6,1,6),plot(nodePos, momG,'bx');
  ylabel('gridMom');
  %axis([0 1 0 7000] )

  %f_name = sprintf('%g.ppm',tstep-1)
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
