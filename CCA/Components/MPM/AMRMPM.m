function KK=amrmpm(problem_type,CFL,NN)

% One dimensional MPM
% bulletproofing

if (~strcmp(problem_type, 'impulsiveBar')  && ...
    ~strcmp(problem_type, 'oscillator') && ...
    ~strcmp(problem_type, 'collidingBars') )
  fprintf('ERROR, the problem type is invalid\n');
  fprintf('     The valid types are impulsiveBar, oscillator, collidingBars\n');
  fprintf('usage:  amrmpm(problem type, cfl, NN)\n');
  return;
end
%__________________________________
% hard coded domain
PPC =1;
E   =1e8;
density = 1000.;

c = sqrt(E/density);

bar_length =1.;
domain     =1.;
area       =1.;
dx         =domain/(NN-1)
volP       =dx/PPC;

dt=CFL*dx/c;

%__________________________________
% region structure
global numRegions;
global Regions;  
numRegions    = int8(1);               % partition the domain into numRegions
Regions       = cell(numRegions,1);    % array that holds the individual region information
R.min         = 0;                     % location of left point
R.max         = 1.0;                   % location of right point
R.refineRatio = 1;
R.dx          = dx
R.NN          = (R.max - R.min)/R.dx
Regions{1}    = R;

R.min         = 0.3;                       
R.max         = 0.6;
R.refineRatio = 1;
R.dx          = dx/R.refineRatio; 
R.NN          = (R.max - R.min)/R.dx                   
Regions{2}    = R;

R.min         = 0.6;                       
R.max         = domain;
R.refineRatio = 1;
R.dx          = dx/R.refineRatio; 
R.NN          = (R.max - R.min)/R.dx       
Regions{3}    = R;

NN = int8(0);
dt = 9999;
for r=1:numRegions
  R = Regions{r};
  NN = NN + R.NN;
  dt = min(dt, CFL*R.dx/c);
  fprintf( 'region %g, min: %g, \t max: %g \t refineRatio: %g dx: %g, dt: %g \n',r, R.min, R.max, R.refineRatio, R.dx, dt)
end

% initialize dx
if(0)
c = 1;
for r=1:numRegions
  R = Regions{r};
  for ig=1:R.NN
    dx(c) = R.dx;
    c = c+1;
  end
end
c
end
NN = NN +1



%__________________________________
% create particles
ip=1;
xp(1)=dx/(2.*PPC);

if ~strcmp(problem_type, 'collidingBars')
  while xp(ip)+dx/PPC < bar_length
    ip = ip+1;
    xp(ip)=xp(ip-1)+dx/PPC;
  end
end

if strcmp(problem_type, 'collidingBars')
  %left bar
  while xp(ip)+dx/PPC < (bar_length/2. - dx)
    ip=ip+1;
    xp(ip)=xp(ip-1)+dx/PPC;
  end
  
  ip=ip+1;
  xp(ip)=domain-dx/(2.*PPC);
  
  % right bar
  while xp(ip)-dx/PPC > (bar_length/2. + dx)
    ip=ip+1;
    xp(ip)=xp(ip-1)-dx/PPC;
  end
end

NP=ip  % Particle count

%__________________________________
% preallocate variables for speed
vol       = zeros(1,NP);
massP     = zeros(1,NP);
velP      = zeros(1,NP);
dp        = zeros(1,NP);
stressP   = zeros(1,NP);
extForceP = zeros(1,NP);
Fp        = zeros(1,NP);

xG        = zeros(1,NN);
massG     = zeros(1,NN);
momG      = zeros(1,NN);
velG      = zeros(1,NN);
vel_nobc_G= zeros(1,NN);
accl_G    = zeros(1,NN);
extForceG = zeros(1,NN);
intForceG = zeros(1,NN);

%__________________________________
% initialize other particle variables
for ip=1:NP
  vol(ip)   = volP;
  massP(ip) = volP*density;
  Fp(ip)    = 1.;
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
  Mass      = 10000.;
  period    = 2.*3.14159/sqrt(E/Mass)
  v0        = 0.5;
  Amp       = v0/(2.*3.14159/period);
  massP(NP) = Mass;                    % last particle masss
  velP(NP)  = v0;                      % last particle velocity
end

if strcmp(problem_type, 'collidingBars')
  period = 4.0*dx/100.;
  for ip=1:NP
    velP(ip) =100.0;
    
    if xp(ip) > .5*bar_length
      velP(ip) = -100.0;
    end
  end
  
  close all;
  plot(xp,velP,'bx');
  hold on;
  p=input('hit return');
end

tfinal=1.0*period;

% create array of nodal locations, only used in plotting
for(ig=1:NN)
  xG(ig) = (ig-1)*dx;
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
%__________________________________
% Main timstep loop
while t<tfinal
  tstep = tstep + 1;
  t = t + dt;
  fprintf('timestep %g, dt = %g, time %g \n',tstep, dt, t)

  % initialize arrays to be zero
  for ig=1:NN
    massG(ig)     =1.e-100;
    velG(ig)      =0.;
    vel_nobc_G(ig)=0.;
    accl_G(ig)    =0.;
    extForceG(ig) =0.;
    intForceG(ig) =0.;
  end
  
  % project particle data to grid
  for ip=1:NP
    [nodes,Ss]=findNodesAndWeights(xp(ip));
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
  [stressP,vol,Fp]=computeStressFromVelocity(xp,dt,velG,E,Fp,volP,NP);

  %compute internal force
  for ip=1:NP
    [nodes,Gs,dx]=findNodesAndWeightGradients(xp(ip));
    for ig=1:2
      intForceG(nodes(ig)) = intForceG(nodes(ig)) - (Gs(ig)/dx) * stressP(ip) * vol(ip);
    end
  end

  %compute the acceleration and new velocity on the grid
  accl_G    =(intForceG + extForceG)./massG;
  vel_new_G = velG + accl_G.*dt;

  %set the velocity BC on the grid
  for ibc=1:numBCs
    vel_new_G(BCNode(ibc)) = BCValue(ibc);
  end

  for ig=1:NN
    accl_G(ig)  = (vel_new_G(ig) - vel_nobc_G(ig))/dt;
  end

  %project changes back to particles
  for ip=1:NP
    [nodes,Ss]=findNodesAndWeights(xp(ip));
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
  
  for ip=1:NP
    KE(tstep) = KE(tstep) + .5*massP(ip) * velP(ip) * velP(ip);
    SE(tstep) = SE(tstep) + .5*stressP(ip) * (Fp(ip)-1.) * vol(ip);
    TE(tstep) = KE(tstep) + SE(tstep);
  end

  %__________________________________
  % compute the exact tip deflection
  if strcmp(problem_type, 'impulsiveBar')
    if(T<=period/2.)
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
  if strcmp(problem_type, 'collidingBars')
    if mod(tstep,100)
     close all;
     %% Create figure
     figure1 = figure;

     %% Create axes
     axes1 = axes('Parent',figure1);
     xlim(axes1,[0 1]);
     box(axes1,'on');
     hold(axes1,'all');
     plot(xp,massP,'bx');
     hold on;
     p=input('hit return');
    end
  end
  
  % bulletproofing
  % particles can't leave the domain
  for ip=1:NP
    if(xp(ip) >= domain) 
      t = tfinal;
      fprintf('\nparticle(%g) position is outside the domain: %g \n',ip,xp(ip))
      fprintf('now exiting the time integration loop\n\n') 
    end
  end

end
%__________________________________
%  plot the results
close all;
set(gcf,'position',[100,100,900,900]);

subplot(3,1,1),plot(TIME,DX_tip,'bx');

hold on;
subplot(3,1,1),plot(TIME,Exact_tip,'r-');

tmp = sprintf('%s, ppc: %g, NN: %g',problem_type, PPC, NN);
title(tmp);
legend('Simulation','Exact')
xlabel('Time [sec]');
ylabel('Tip Deflection')

subplot(3,1,2),plot(TIME,TE,'b-');
ylabel('Total Energy');

E_err=TE(1)-TE(tstep)

% compute error
err=abs(DX_tip-Exact_tip);

subplot(3,1,3),plot(TIME,err,'b-');
ylabel('Abs(error)')

print ('-dpng', problem_type)
length(TIME)

%__________________________________
%  write the results out to files
fid = fopen('particleData.dat', 'w');
fprintf(fid,'%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
fprintf(fid,'p, massP, velP, stressP, extForceP, Fp\n');
for ip=1:NP
  fprintf(fid,'%g, %g, %g, %g, %g, %g\n',ip, massP(ip), velP(ip), stressP(ip), extForceP(ip), Fp(ip));
end
fclose(fid);

fid = fopen('gridData.dat', 'w');
fprintf(fid,'%s, PPC: %g, NN %g\n',problem_type, PPC, NN);
fprintf(fid,'g, xG, massG, velG, extForceG, intForceG, accl_G\n');
for ig=1:NN
  fprintf(fid,'%g, %g, %g, %g, %g, %g %g\n',ig, xG(ig), massG(ig), velG(ig), extForceG(ig), intForceG(ig), accl_G(ig));
end
fclose(fid);

return;


%______________________________________________________________________
% functions
function[node, dx]=positionToNode(xp);
  dx = -9;
  global numRegions
  global Regions
  
  for r=1:numRegions
    R = Regions{r};
    if (xp >= R.min & xp <= R.max)
      dx = R.dx;
    end
  end
  
  node = xp/dx;
  node = floor(node) + 1;
  
return;

%__________________________________
function [nodes,Ss]=findNodesAndWeights(xp);
 
% find the nodes that surround the given location and
% the values of the shape functions for those nodes
% Assume the grid starts at x=0.

%node = xp/dx;
%node = floor(node)+1;
  
[node, dx]=positionToNode(xp);  
  
nodes(1)= node;
nodes(2)= nodes(1)+1;
 
dnode = double(node);
 
locx  = (xp-dx*(dnode-1))/dx;
Ss(1) = 1-locx;
Ss(2) = locx;
 
return;

%__________________________________
function [nodes,Gs, dx]=findNodesAndWeightGradients(xp);
 
% find the nodes that surround the given location and
% the values of the shape functions for those nodes
% Assume the grid starts at x=0.
 
%node = xp/dx;
%node = floor(node)+1;
 
[node, dx]=positionToNode(xp);
 
nodes(1) = node;
nodes(2) = nodes(1)+1;
 
dnode = double(node);
 
%locx=(xp-dx*(dnode-1))/dx;
Gs(1) = -1;
Gs(2) = 1;
 
return;

%__________________________________
function [stressP,vol,Fp]=computeStressFromVelocity(xp,dt,velG,E,Fp,volP,NP);
                                                                                
for ip=1:NP
  [nodes,Gs,dx]=findNodesAndWeightGradients(xp(ip));
  gUp=0;
  for ig=1:2
    gUp = gUp + velG(nodes(ig)) * (Gs(ig)/dx);
  end
  
  dF          =1. + gUp * dt;
  Fp(ip)      = dF * Fp(ip);
  stressP(ip) = E * (Fp(ip)-1);
  vol(ip)     = volP * Fp(ip);
end
