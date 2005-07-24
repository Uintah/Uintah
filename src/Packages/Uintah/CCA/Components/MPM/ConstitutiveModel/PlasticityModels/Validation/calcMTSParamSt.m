function calcMTSParamSt

  %
  %  Compute Fisher plot data
  %
  computeFisherPlotData

function computeFisherPlotData

  load StDataRc30Ep0.dat
  dat = StDataRc30Ep0;
  [C30] = computeFisherPlot(dat, 'StDataFisherRc30Ep0.dat');

  %load StDataRc32Ep0.dat
  %dat = StDataRc32Ep0;
  %[C32] = computeFisherPlot(dat, 'StDataFisherRc32Ep0.dat');

  load StDataRc38Ep0.dat
  dat = StDataRc38Ep0;
  [C38] = computeFisherPlot(dat, 'StDataFisherRc38Ep0.dat');

  load StDataRc45Ep0.dat
  dat = StDataRc45Ep0;
  [C45] = computeFisherPlot(dat, 'StDataFisherRc45Ep0.dat');

  load StDataRc49Ep0.dat
  dat = StDataRc49Ep0;
  [C49] = computeFisherPlot(dat, 'StDataFisherRc49Ep0.dat');

  %
  % HY-100 fit
  %
  %sig_a = 50.0e6;
  %edot_0i = 1.0e13;
  %p_i = 0.5;
  %q_i = 1.5;

  %
  % Best fit
  %
  sig_a = 150.0e6;
  edot_0i = 1.0e8;
  p_i = 2/3;
  q_i = 1.0;

  rho0 = 7830.0;
  P = calcP(rho0, rho0, 0, 0);
  Tm = calcTm(rho0, rho0);
  mu_0 = calcmu(rho0, rho0, Tm, P, 0);

  Rc(1) = 30;
  sigma_i_mu0_pi = C30(2);
  oog0i_qi = -C30(1)/C30(2);
  sigma_i_mu0 = sigma_i_mu0_pi^(1/p_i);
  oog0i = oog0i_qi^q_i;
  sigma_i(1) = sigma_i_mu0*mu_0;
  g0i(1) = 1/oog0i;
  
  Rc(2) = 38;
  sigma_i_mu0_pi = C38(2);
  oog0i_qi = -C38(1)/C38(2);
  sigma_i_mu0 = sigma_i_mu0_pi^(1/p_i);
  oog0i = oog0i_qi^q_i;
  sigma_i(2) = sigma_i_mu0*mu_0;
  g0i(2) = 1/oog0i; 
  
  Rc(3) = 45;
  sigma_i_mu0_pi = C45(2);
  oog0i_qi = -C45(1)/C45(2);
  sigma_i_mu0 = sigma_i_mu0_pi^(1/p_i);
  oog0i = oog0i_qi^q_i;
  sigma_i(3) = sigma_i_mu0*mu_0;
  g0i(3) = 1/oog0i; 
  
  Rc(4) = 49;
  sigma_i_mu0_pi = C49(2);
  oog0i_qi = -C49(1)/C49(2);
  sigma_i_mu0 = sigma_i_mu0_pi^(1/p_i);
  oog0i = oog0i_qi^q_i;
  sigma_i(4) = sigma_i_mu0*mu_0;
  g0i(4) = 1/oog0i; 
  
  for i=1:4
    [Rc(i) sigma_i(i) g0i(i)]
  end
  [pSigma_i] = polyfit(Rc, sigma_i, 3)
  [pg0_i] = polyfit(Rc, g0i, 3)

  RcMin = 25;
  RcMax = 50;
  nRc = 100;
  dRc = (RcMax-RcMin)/nRc;
  for i=1:nRc+1
   Rcc(i) = RcMin + (i-1)*dRc;
   sig_i(i) = 0.243645*Rcc(i)^3 - 32.518*Rcc(i)^2 + 1452.456*Rcc(i) ...
              - 20121.018;
   sig_i(i) = sig_i(i)*1.0e6;
   g0_i(i) = -0.0016694*Rcc(i)^3 + 0.22814*Rcc(i)^2 - 10.220*Rcc(i) ...
              + 150.8908;
  end

  figure
  subplot(1,2,1);
  plot(Rc, sigma_i, 'ro'); hold on;
  plot(Rcc, sig_i, 'b-');
  subplot(1,2,2);
  plot(Rc, g0i, 'go'); hold on;
  plot(Rcc, g0_i, 'b-');

  %=========================================================================

function [C] = computeFisherPlot(dat, fileName)

  fig = figure;

  dat = sortrows(dat, 3);
  edot = dat(:,2);
  T = dat(:,3);
  sig_y = dat(:,6);
  rho = 7830.0;
  rho0 = 7830.0;
  k = 1.3806503e-23;
  b = 2.48e-10;

  %
  % HY-100 fit
  %
  %sig_a = 50.0e6;
  %edot_0i = 1.0e13;
  %p_i = 0.5;
  %q_i = 1.5;

  %
  % Best fit
  %
  sig_a = 150.0e6;
  edot_0i = 1.0e8;
  p_i = 2/3;
  q_i = 1.0;

  [n,m] = size(dat);
  for i=1:n
    P = calcP(rho, rho0, T(i), T(i));
    Tm = calcTm(rho, rho0);
    mu(i) = calcmu(rho, rho0, Tm, P, T(i));
    xx(i) = (k*T(i)/(mu(i)*b^3)*log(edot_0i/edot(i)))^(1/q_i);
    yy(i) = ((sig_y(i)*1.0e6 - sig_a)/mu(i))^p_i;
    str = sprintf('(%g,  %g,  %g)',i,edot(i),T(i));
    text(xx(i)+0.005,yy(i),str); hold on;
  end
  plot(xx, yy, 'rs'); hold on;
  [C,S] = polyfit(xx,yy,1);
  xmin = 0.0;
  xmax = 0.5;
  nx = 100;
  dx = xmax/nx;
  for i=1:nx+1
    xfit(i) = xmin + (i-1)*dx;
    yfit(i) = C(2) + C(1)*xfit(i);
  end
  plot(xfit, yfit, 'b-'); hold on;
  fid = fopen(fileName,'w');
  for i=1:n
    fprintf(fid,'%g %g %g %g %g %g\n', ...
            xx(i), yy(i), T(i), edot(i), sig_y(i), mu(i)/1.0e6);
  end
  fclose(fid);

function [p] = calcP(rho, rho0, T, T0)

  % Data from Brown and Gust 79
  eta = rho/rho0;
  C0 = 3935.0;
  S_alpha = 1.578;
  Gamma0 = 1.69;

  Cv = calcCp(T);
  zeta = rho/rho0 - 1;
  E = Cv*(T-T0)*rho0;

  if (rho == rho0)
    p = Gamma0*E;
  else
    numer = rho0*C0^2*(1/zeta + 1 - 0.5*Gamma0);
    denom = 1/zeta + 1 - S_alpha;
    p = numer/denom^2 + Gamma0*E;
  end

function [Cp] = calcCp(T)

  Tc = 1040.0;
  if (T == Tc)
    T = T - 1.0;
  end
  if (T < Tc) 
    t = 1 - T/Tc;
    A = 190.14;
    B = -273.75;
    C = 418.30;
    n = 0.2;
    Cp = A + B*t + C/t^n;
  else
    t = T/Tc - 1.0;
    A = 465.21;
    B = 267.52;
    C = 58.16;
    n = 0.35;
    Cp = A + B*t + C/t^n;
  end

function [Tm] = calcTm(rho, rho0)

  %
  % Constants and derivative from Guinan and Steinberg, 1974
  %
  B0 = 1.66e11; 
  dB_dp0 = 5.29;
  G0 = 0.819e11;
  dG_dp0 = 1.8;

  %
  % Calculate the pressure 
  %
  eta = rho/rho0;
  p = calcP(rho, rho0, 300, 300);

  %
  % BPS parameters for Fe at T = 300K and p = 0
  %
  kappa = 1;  %Screw dislocation
  z = 8.0; % bcc lattice
  b2rhoTm = 0.64;
  b2rhoTm = b2rhoTm+0.14;
  alpha = 2.9;
  lambda = 1.30; % bcc lattice
  a = 5.4057*0.53e-10;
  vws = a^3/2; % bcc crystal
  kbTm = (kappa*lambda*vws*G0)/(8.0*pi*log(z-1.0))*log(alpha^2/(4.0*b2rhoTm));
  kb = 1.3806503e-23;
  Tm0 = kbTm/kb; 

  %
  % Calculate the bulk and shear factors
  %
  Bfac = 1.0 + dB_dp0/B0*p;
  Gfac = dG_dp0/G0*p;

  eta = Bfac^(1/dB_dp0);
  %
  % Calculate Tm at pressure p
  %
  Tm = Tm0/eta*(1.0 + Gfac/eta^(1/3));

function [mu] = calcmu(rho, rho0, Tm, P, T)

  mu0 = 90.0e9;
  zeta = 0.04;
  dmu_dp = 1.8;
  dmu_dp_mu0 = dmu_dp/mu0;
  C = 0.080;
  m = 55.947;
  k = 1.38e4/1.6605402;

  That = T/Tm;
  if (That < 1+zeta)
    denom = zeta*(1 - That/(1+zeta));
    t0 = 1 - That;
    J = 1 + exp(-t0/denom);
    eta = (rho/rho0)^(1/3);
    t1 = mu0*(1 + dmu_dp_mu0*P/eta);
    t2 = rho*k*T/(C*m);
    mu = (t0*t1 + t2)/J;
  else
    mu = 0;
  end

