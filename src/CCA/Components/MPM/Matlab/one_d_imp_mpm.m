function odmn=one_d_mpm_new(problem_type,CFL,NN)

% One dimensional MPM

PPC=1;

E=1e8;
density = 1000.;
disp_crit=1.e-4;
energy_crit=1.e-3;

c = sqrt(E/density);

bar_length = 1.;
domain=1.;
area=1.;
dx=domain/(NN-1);
volp=dx/PPC;

dt=CFL*(1./(NN-1))/c;

if problem_type==1  %impulsively loaded bar
    period = sqrt(16.*bar_length*bar_length*density/E)
    TipForce=10.;
    D=TipForce*bar_length/(area*E);
    M=4.*D/period;
end
if problem_type==2 %simple oscillator
    Mass=10000.;
    period=2.*3.14159/sqrt(E/Mass);
    v0=1.0;
    Amp=v0/(2.*3.14159/period);
end
if problem_type==5 % colliding bars
    period=10.0*dx/100.;
end

tfinal=1.0*period;

% create particles
ip=1;
                                                                                
if problem_type~=5
  xp(ip)=dx/(2.*PPC);
                                                                                
  while xp(ip)+dx/PPC < bar_length
     ip=ip+1;
     xp(ip)=xp(ip-1)+dx/PPC;
  end
end

if problem_type==5
  xp(ip)=dx/(2.*PPC);
  while xp(ip)+dx/PPC < (bar_length/2. - dx)
     ip=ip+1;
     xp(ip)=xp(ip-1)+dx/PPC;
  end
  ip=ip+1;
  xp(ip)=domain-dx/(2.*PPC);
  while xp(ip)-dx/PPC > (bar_length/2. + dx)
     ip=ip+1;
     xp(ip)=xp(ip-1)-dx/PPC;
  end
end

NP=ip  % Particle count

% initialize other particle variables
for ip=1:NP
    vol(ip)=volp;
    mp(ip)=volp*density;
    vp(ip)=0.;
    ap(ip)=0.;
    dp(ip)=0.;
    sigp(ip)=0.;
    Fp(ip)=1.;
    exFp(ip)=0.;
end

if problem_type==1
   exFp(NP)=TipForce;
end
if problem_type==2
   mp(NP)=Mass;
   vp(NP)=v0;
end
if problem_type==5
 for ip=1:NP
   if xp(ip) < .5*bar_length
      vp(ip)=100.0;
   end
   if xp(ip) > .5*bar_length
      vp(ip)=-100.0;
   end
 end
 close all;
 plot(xp,mp,'bx');
 hold on;
 p=input('hit return');
end


% create array of nodal locations, only used in plotting
for(ig=1:NN)
    xg(ig)=(ig-1)*dx;
    dug(ig)=0.;
end

% set up BCs
numBCs=1;
if problem_type==5
    numBCs=0;
end

BCNode(1)=1;
BCValue(1)=0.;

t=0.;
tstep=0;

while t<tfinal
    tstep=tstep+1;
    t=t+dt;
    if problem_type==3
        exFp=mp.*t;
    end
    if problem_type==4
        exFp=mp;
    end

    % initialize arrays to be zero
    for ig=1:NN
        mg(ig)=1.e-100;
        vg(ig)=0.;
        ug(ig)=0.;
        dug(ig)=0.;
        vg_nobc(ig)=0.;
        ag(ig)=0.;
        Feg(ig)=0.;
        Fig(ig)=0.;
    end

    % project particle data to grid
    for ip=1:NP
        [nodes,Ss]=findNodesAndWeights(xp(ip),dx);
        for ig=1:2
            mg(nodes(ig))=mg(nodes(ig))+mp(ip)*Ss(ig);
            vg(nodes(ig))=vg(nodes(ig))+mp(ip)*vp(ip)*Ss(ig);
            Feg(nodes(ig))=Feg(nodes(ig))+exFp(ip)*Ss(ig);
            ag(nodes(ig))=ag(nodes(ig))+mp(ip)*ap(ip)*Ss(ig);
        end
    end

    % normalize by the mass
    vg=vg./mg;
    vg_nobc=vg;
    ag=ag./mg;
    
    iter=0;
    dispnorm_max=0.;
    energynorm_0=0.;
    converged=0;

    while converged ~=1
      iter=iter+1;
      %compute particle stress
      [sigp,vol,Fp]=computeStressFromDisplacementInc(xp,dx,ug,E,Fp,volp,NP);

      %zero out internal force vector and stiffness matrix
      for ig=1:NN
        Fig(ig)=0.;
        for jg=1:NN
          KK(ig,jg)=0.;
        end
      end

      %compute internal force
      for ip=1:NP
          [nodes,Gs]=findNodesAndWeightGradients(xp(ip),dx);
          for ig=1:2
              Fig(nodes(ig))=Fig(nodes(ig))-(Gs(ig)/dx)*sigp(ip)*vol(ip);
          end
      end

      %assemble RHS
      Q=Feg+Fig-mg.*((4./(dt*dt))*ug-(4./dt)*vg-ag);

      %assemble stiffness
      for ip=1:NP
          [nodes,Gs]=findNodesAndWeightGradients(xp(ip),dx);
          for ig=1:2
            for jg=1:2
              KK(nodes(ig),nodes(jg))=KK(nodes(ig),nodes(jg))+(Gs(ig)/dx)*(Gs(jg)/dx)*E*vol(ip);
            end
          end
      end

      %add inertial term
      for(ig=1:NN)
         KK(ig,ig)=KK(ig,ig)+(4./(dt*dt))*mg(ig);
         if KK(ig,ig)<1e-10
            for jg=1:NN
              KK(jg,ig)=0.;
            end
            KK(ig,ig)=1.;
            Q(ig)=0.;
         end
      end

      %apply boundary conditions
      for ibc=1:numBCs
        Q(BCNode(ibc))=BCValue(ibc);
        for ig=1:NN
          KK(ig,BCNode(ibc))=0.;
        end
        KK(BCNode(ibc),BCNode(ibc))=1.;
      end

      %compute displacement increment
      dug=Q/KK;

      %update kinematics on grid
      ug=ug+dug;

      %check for convergence
      dispnorm=norm(dug);
      energynorm=norm(dug.*Q);

      if dispnorm>dispnorm_max
       dispnorm_max=dispnorm;
      end
      if iter==1
       energynorm_max=energynorm;
      end

      if dispnorm/dispnorm_max < disp_crit
        if energynorm/energynorm_max < energy_crit
          converged=1;
        end
      end

%      dispnorm/dispnorm_max
%      energynorm/energynorm_max

%p=input('hit return');

    end %while
    
    %compute particle stress based on converged state
    [sigp,vol,Fp]=computeStressFromDisplacementFinal(xp,dx,ug,E,Fp,volp,NP);

    %compute acceleration
    ag=(4./(dt*dt))*ug-(4./dt)*vg-ag;

    %interpolate changes back to particles
    for ip=1:NP
        [nodes,Ss]=findNodesAndWeights(xp(ip),dx);
        dxp=0.;
        acc=0.;
        for ig=1:2
            dxp=dxp+ug(nodes(ig))*Ss(ig);
            acc=acc+ag(nodes(ig))*Ss(ig);
        end
        dvp=(ap(ip)+acc)*0.5*dt;
        ap(ip)=acc;
        vp(ip)=vp(ip)+dvp;
        xp(ip)=xp(ip)+dxp;
        dp(ip)=dp(ip)+dxp;
    end

    DX_tip(tstep)=dp(NP);
    T=t;%-dt;

    KE(tstep)=0.;SE(tstep)=0.;
    for ip=1:NP
        KE(tstep)=KE(tstep)+.5*mp(ip)*vp(ip)*vp(ip);
        SE(tstep)=SE(tstep)+.5*sigp(ip)*(Fp(ip)-1.)*vol(ip);
        TE(tstep)=KE(tstep)+SE(tstep);
    end

    if problem_type==1
        if(T<=period/2.)
            Exact_tip(tstep)=M*T;
        else
            Exact_tip(tstep)=4.*D-M*T;
        end
    end
    if problem_type==2
       Exact_tip(tstep)=Amp*sin(2.*3.14159*T/period);
    end

    TIME(tstep)=t;

    if problem_type==5
     if mod(tstep,100)
      close all;
      %% Create figure
      figure1 = figure;
                                                                                
      %% Create axes
      axes1 = axes('Parent',figure1);
      xlim(axes1,[0 1]);
      box(axes1,'on');
      hold(axes1,'all');
      plot(xp,mp,'bx');
      hold on;
      p=input('hit return');
     end
    end
end

if problem_type ~= 5
  close all;
  subplot(2,1,1),plot(TIME,DX_tip,'bx');
  hold on;
  subplot(2,1,1),plot(TIME,Exact_tip,'r-');
  subplot(2,1,2),plot(TIME,TE,'b-');

  E_err=TE(1)-TE(tstep)

  % compute error
  err=abs(DX_tip(tstep)-Exact_tip(tstep))
end

length(TIME)

return;

function [nodes,Ss]=findNodesAndWeights(xp,dx);
 
% find the nodes that surround the given location and
% the values of the shape functions for those nodes
% Assume the grid starts at x=0.
 
node = xp/dx;
node=floor(node)+1;
 
nodes(1)= node;
nodes(2)=nodes(1)+1;
 
dnode=double(node);
 
locx=(xp-dx*(dnode-1))/dx;
Ss(1)=1-locx;
Ss(2)=locx;
 
return;

function [nodes,Gs]=findNodesAndWeightGradients(xp,dx);
 
% find the nodes that surround the given location and
% the values of the shape functions for those nodes
% Assume the grid starts at x=0.
 
node = xp/dx;
node=floor(node)+1;
 
nodes(1)= node;
nodes(2)=nodes(1)+1;
 
dnode=double(node);
 
%locx=(xp-dx*(dnode-1))/dx;
Gs(1)=-1;
Gs(2)=1;
 
return;

function [sigp,vol,Fp]=computeStressFromDisplacementInc(xp,dx,ug,E,Fp,volp,NP);
 
for ip=1:NP
    [nodes,Gs]=findNodesAndWeightGradients(xp(ip),dx);
    gUp=0;
    for ig=1:2
        gUp=gUp+ug(nodes(ig))*(Gs(ig)/dx);
    end
    dF=1.+gUp;
    F=dF*Fp(ip);
    sigp(ip)=E*(F-1.);
    vol(ip)=volp*F;
end

function [sigp,vol,Fp]=computeStressFromDisplacementFinal(xp,dx,ug,E,Fp,volp,NP);
for ip=1:NP
    [nodes,Gs]=findNodesAndWeightGradients(xp(ip),dx);
    gUp=0;
    for ig=1:2
        gUp=gUp+ug(nodes(ig))*(Gs(ig)/dx);
    end
    dF=1.+gUp;
    Fp(ip)=dF*Fp(ip);
    sigp(ip)=E*(Fp(ip)-1.);
    vol(ip)=volp*Fp(ip);
end
