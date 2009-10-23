%Reference:  Jin Ma, Honbing Lu,and Ranga Komanduri, "Structured Mesh
% Refinement in Generalized Interpolation Material Point (GIMP) Method
% for Simulation of Dynamic Problems," CMES, vol. 12, no.3, pp. 213-227, 2006

%______________________________________________________________________



function KK=amrmpm(problem_type,CFL,R1_dx)

%  valid options:
%  problem type:  impulsiveBar, oscillator, collidingBars
%  CFL:            0 < cfl < 1, usually 0.2 or so.
%  R1_dx:          cell spacing in region 1.



% One dimensional MPM
% bulletproofing

if (~strcmp(problem_type, 'impulsiveBar')  && ...
    ~strcmp(problem_type, 'oscillator') && ...
    ~strcmp(problem_type, 'collidingBars') )
  fprintf('ERROR, the problem type is invalid\n');
  fprintf('     The valid types are impulsiveBar, oscillator, collidingBars\n');
  fprintf('usage:  amrmpm(problem type, cfl, R1_dx)\n');
  return;
end
%__________________________________
% hard coded domain
PPC =1;
E   =1e8;
density = 1000.;
BigNum  = int8(1e4);
c = sqrt(E/density);

bar_length =1.;
domain     =1.;
area       =1.;
plotSwitch = 0;

% HARDWIRED FOR TESTING
NN          = 16;
R1_dx       =domain/(NN-1)

if (mod(domain,R1_dx) ~= 0)
  fprintf('ERROR, the dx in Region 1 does not divide into the domain evenly\n');
  fprintf('usage:  amrmpm(problem type, cfl, R1_dx)\n');
  return;
end

%__________________________________
% region structure 
numRegions    = int8(3);               % partition the domain into numRegions
Regions       = cell(numRegions,1);    % array that holds the individual region information

R.min         = 0;                     % location of left point
R.max         = 1/3;                   % location of right point
R.refineRatio = 1;
R.dx          = R1_dx;
R.volP        = R.dx/PPC;
R.NN          = int8( (R.max - R.min)/R.dx );
Regions{1}    = R;

R.min         = 1/3;                       
R.max         = 2/3;
R.refineRatio = 1;
R.dx          = R1_dx/R.refineRatio;
R.volP        = R.dx/PPC;
R.NN          = int8( (R.max - R.min)/R.dx );                  
Regions{2}    = R;

R.min         = 2/3;                       
R.max         = domain;
R.refineRatio = 1;
R.dx          = R1_dx/R.refineRatio; 
R.volP        = R.dx/PPC;
R.NN          = int8( (R.max - R.min)/R.dx + 1);       
Regions{3}    = R;

NN = int8(0);
dt = 9999;

for r=1:numRegions
  R = Regions{r};
  NN = NN + R.NN;
  dt = min(dt, CFL*R.dx/c);
  fprintf( 'region %g, min: %g, \t max: %g \t refineRatio: %g dx: %g, dt: %g NN: %g\n',r, R.min, R.max, R.refineRatio, R.dx, dt, R.NN)
end
NN
%__________________________________
% compute the zone of influence
% compute the positions of the nodes
Lx      = zeros(NN,2);
nodePos = zeros(NN,1);      % node Position
nodeNum=1;

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

nn = 1;
for r=1:numRegions
  R = Regions{r};
  fprintf('-------------------------Region %g\n',r);
  for n=1:R.NN
    fprintf( 'Node:  %g, nodePos: %6.5f, \t Lx(1): %6.5f Lx(2): %6.5f\n',nn, nodePos(nn),Lx(nn,1), Lx(nn,2));
    nn = nn + 1;
  end
end
input('hit return')


%__________________________________
% create particles
ip=1;
xp   = zeros(1,BigNum);
xp(1)=R1_dx/(2.*PPC);

for r=1:numRegions
  R = Regions{r};
  
  dx_P = R.dx/PPC;                             % particle dx
  
  if ~strcmp(problem_type, 'collidingBars')    % Everything except the collingBar
    while (xp(ip) + dx_P > R.min ) && ...
          (xp(ip) + dx_P < R.max ) && ...
          (xp(ip) + dx_P < bar_length)
          
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

NP=ip  % number of particles

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

xG        = zeros(NN,1);
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


%__________________________________
% initialize other particle variables
for ip=1:NP
  [volP]    = positionToVolP(xp(ip), numRegions, Regions);
  vol(ip)   = volP;
  massP(ip) = volP*density;
  Fp(ip)    = 1.;                     % total deformation
end

%__________________________________
% Problem dependent parameters
if strcmp(problem_type, 'impulsiveBar')
  period        = sqrt(16.*bar_length*bar_length*density/E);
  TipForce      = 10.;
  D             = TipForce*bar_length/(area*E);
  M             = 4.*D/period;
  extForceP(NP) = TipForce;
end

if strcmp(problem_type, 'oscillator')
  Mass          = 10000.;
  period        = 2.*3.14159/sqrt(E/Mass);
  v0            = 0.5;
  Amp           = v0/(2.*3.14159/period);
  massP(NP)     = Mass;                    % last particle masss
  velP(NP)      = v0;                      % last particle velocity
end

if strcmp(problem_type, 'collidingBars')
  period = 4.0*dx/100.;
  for ip=1:NP
    velP(ip) =100.0;
    
    if xp(ip) > .5*bar_length
      velP(ip) = -100.0;
    end
  end
  
end

tfinal=1.0*period;

% create array of nodal locations, only used in plotting
ig = 1;
for r=1:numRegions
  R = Regions{r};
  for  c=1:R.NN
    xG(ig) = (ig-1)*R.dx;
    ig = ig + 1;
  end
end
% set up BCs
numBCs=1;

if strcmp(problem_type, 'collidingBars')
  numBCs=0;
end

BCNode(1)  = 1;
BCNode(2)  = NN;
BCValue(1) = 0.;
BCValue(2) = 1.;

t = 0.0;
tstep = 0;

%plot initial conditions
%plotResults(t,xp,dp,velP)


%==========================================================================
% Main timstep loop
while t<tfinal
  tstep = tstep + 1;
  t = t + dt;
  %fprintf('timestep %g, dt = %g, time %g \n',tstep, dt, t)

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
  % project particle data to grid
  for ip=1:NP
    [nodes,Ss]=findNodesAndWeights(xp(ip), numRegions, Regions, nodePos, Lx);
    for ig=1:2
      massG(nodes(ig))     = massG(nodes(ig))     + massP(ip) * Ss(ig);
      velG(nodes(ig))      = velG(nodes(ig))      + massP(ip) * velP(ip) * Ss(ig);
      extForceG(nodes(ig)) = extForceG(nodes(ig)) + extForceP(ip) * Ss(ig);
    end
  end

  % normalize by the mass
  velG = velG./massG;
  vel_nobc_G = velG;

  % set velocity BC
  for ibc=1:numBCs
    velG(BCNode(ibc)) = BCValue(ibc);
  end

  %compute particle stress
  [stressP,vol,Fp]=computeStressFromVelocity(xp,dt,velG,E,Fp,NP,numRegions, Regions);

  %compute internal force
  for ip=1:NP
    [nodes,Gs,dx]=findNodesAndWeightGradients(xp(ip),numRegions, Regions);
    for ig=1:2
      intForceG(nodes(ig)) = intForceG(nodes(ig)) - Gs(ig) * stressP(ip) * vol(ip);
    end
  end

  %compute the acceleration and new velocity on the grid
  accl_G    =(intForceG + extForceG)./massG;
  vel_new_G = velG + accl_G.*dt;

  %set velocity BC
  for ibc=1:numBCs
    vel_new_G(BCNode(ibc)) = BCValue(ibc);
  end

  % compute the acceleration on the grid
  for ig=1:NN
    accl_G(ig)  = (vel_new_G(ig) - vel_nobc_G(ig))/dt;
  end
  
  %__________________________________
  %project changes back to particles
  for ip=1:NP
    [nodes,Ss]=findNodesAndWeights(xp(ip),numRegions, Regions, nodePos, Lx);
    dvelP = 0.;
    dxp   = 0.;
    
    for ig=1:2
      dvelP = dvelP + accl_G(nodes(ig))    * dt * Ss(ig);
      dxp   = dxp   + vel_new_G(nodes(ig)) * dt * Ss(ig);
    end
    
    velP(ip) = velP(ip) + dvelP;
    xp(ip)   = xp(ip) + dxp;
    dp(ip)   = dp(ip) + dxp;
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
  % compute the exact tip deflection
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

  TIME(tstep)=t;
  %__________________________________
  % plot intantaneous solution
  if (mod(tstep,100) == 0) && (plotSwitch == 1)
    plotResults(t, xp,dp,velP)
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
plotFinalResults(TIME, DX_tip, Exact_tip, TE, problem_type, PPC, NN)

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
  fprintf(fid,'%16.15f, %16.15f, %16.15f, %16.15f, %16.15f, %16.15f %16.15f\n',ig, xG(ig), massG(ig), velG(ig), extForceG(ig), intForceG(ig), accl_G(ig));
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
  dx = -9;
  
  for r=1:numRegions
    R = Regions{r};
    if ((xp >= R.min) && (xp < R.max))
      dx = R.dx;
    end
  end
  
  node = xp/dx;
  node = floor(node) + 1;
end
%__________________________________
%
function[volP]=positionToVolP(xp, numRegions, Regions)
  volP = -9;
 
  for r=1:numRegions
    R = Regions{r};
    if ((xp >= R.min) && (xp < R.max))
      volP = R.dx;
      return;
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

  % old method of computing the shape function
  locx  = (xp-dx*(dnode-1))/dx;
  Ss_old_1 = 1-locx;
  Ss_old_2 = locx;
  
  % bulletproofing
  diff1 = abs(Ss(1) - Ss_old_1);
  diff2 = abs(Ss(2) - Ss_old_2);
  
  if (diff1 > 1e-14  || diff2 > 1e-14)
    fprintf('There is a problem with one of the shape functions, node %g, delX: %g \n',node, delX);
    fprintf(' Ss(1):  %16.14f , Ss_old:  %16.14f, diff %16.14f \n', Ss(1), Ss_old_1, diff1);
    fprintf(' Ss(2):  %16.14f , Ss_old:  %16.14f, diff %16.14f \n', Ss(2), Ss_old_2, diff1);
  end
  
end


%__________________________________
function [nodes,Ss]=findNodesAndWeights_gimp(xp, numRegions, Regions, nodePos, Lx)
 
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 12 of the reference

  [node, dx]=positionToNode(xp, numRegions, Regions);  

  nodes(1)= node;
  nodes(2)= node+1;
  
  Lx_minus = dx;
  Lx_plus  = dx;
  lp       = dx/2;          % hardwired for now
  
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
    
    Ss(1) = (t1 + t2)/(2.0*lp);
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
function [nodes,Ss]=findNodesAndWeights_old(xp, numRegions, Regions, nodePos, Lx)
 
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.

  [node, dx]=positionToNode(xp, numRegions, Regions);  

  nodes(1)= node;
  nodes(2)= nodes(1)+1;

  dnode = double(node);

  locx  = (xp-dx*(dnode-1))/dx;
  Ss(1) = 1-locx;
  Ss(2) = locx;
end

%__________________________________
function [nodes,Gs, dx]=findNodesAndWeightGradients(xp, numRegions, Regions);
 
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
function [stressP,vol,Fp]=computeStressFromVelocity(xp,dt,velG,E,Fp,NP, numRegions, Regions)
                                                                                
  for ip=1:NP
    [nodes,Gs,dx] = findNodesAndWeightGradients(xp(ip), numRegions, Regions);
    [volP]        = positionToVolP(xp(ip), numRegions, Regions);
    gUp=0;
    for ig=1:2
      gUp = gUp + velG(nodes(ig)) * Gs(ig);
    end

    dF          =1. + gUp * dt;
    Fp(ip)      = dF * Fp(ip);
    stressP(ip) = E * (Fp(ip)-1);
    vol(ip)     = volP * Fp(ip);
  end
end

%__________________________________
function plotResults(t, xp, dp, velP)

  close all;
  set(gcf,'position',[100,100,900,900]);

  % particle velocity vs position
  subplot(2,1,1),plot(xp,velP,'bx');

  tmp = sprintf('Time %g',t);
  title(tmp);
  xlabel('Particle Position');
  ylabel('velocity')

  % particle ???? vs position
  subplot(2,1,2),plot(xp,dp,'b-');
  ylabel('dp');
  input('hit return');
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

