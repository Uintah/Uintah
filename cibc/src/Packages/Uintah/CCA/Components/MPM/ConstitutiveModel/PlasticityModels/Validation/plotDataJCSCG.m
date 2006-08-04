function plotDataJCSCGSt

  clear all; close all;
  %color = ['rgbmkrgbmkrgbmk'];
  %ep = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5 1.7 2.0];
  %epdot = [1.0e-6 1.0e-5 1.0e-4 1.0e-3 1.0e-2 1.0e-1 1.0 1.0e1 1.0e2 1.0e3 1.0e4 ...
  %         1.0e5 1.0e6 1.0e7 1.0e8];
  ep    = [0.00   0.25   0.50   0.75  1.00  1.25  1.50  1.75  2.0];
  epdot = [1.0e-6 1.0e-4 1.0e-2 1.0e0 1.0e1 1.0e2 1.0e4 1.0e6 1.0e8];
  T =     [300    500    700    900   1100   1300   1500   1600   1700];

  %
  % Compute JC data for stress vs temp (at various strains)
  %
  [n, m] = size(T);
  for i=1:m
    epdot_1 = 1.0;
    for j=1:m
      sig_JC_1(i,j) = JC(epdot_1, ep(j), T(i));
      sig_SCG_1(i,j) = SCG(epdot_1, ep(j), T(i));
    end
  end
  fig1 = figure;
  set(fig1, 'Position', [412 313 915 632]);
  fac = 1/m;
  for j=1:m
    p1(j) = plot(T, sig_JC_1(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(T, sig_SCG_1(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
    set(p1(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    set(p2(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    str_ep(j,:) = sprintf('\\epsilon = %4s',num2str(ep(j)));
  end
  tt = title('Strain Rate = 1.0/s, various \epsilon_p');
  xt = xlabel('T (K)');
  yt = ylabel('\sigma_y (MPa)');
  set([tt xt yt], 'FontName', 'bookman', 'FontSize', 16);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 16);
  legend(p1,str_ep,-1);

  %
  % Compute JC data for stress vs temp (at various strain rates)
  %
  [n, m] = size(T);
  for i=1:m
    ep_2 = 0.3;
    for j=1:m
      sig_JC_2(i,j) = JC(epdot(j), ep_2, T(i));
      sig_SCG_2(i,j) = SCG(epdot(j), ep_2, T(i));
    end
  end
  fig2 = figure;
  set(fig2, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(T, sig_JC_2(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(T, sig_SCG_2(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
    set(p1(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    set(p2(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    str_epdot(j,:) = sprintf('d\\epsilon/dt = %5s/s',num2str(epdot(j),'%3.0e'));
  end
  tt = title('\epsilon_p = 0.3, various strain rates');
  xt = xlabel('T (K)');
  yt = ylabel('\sigma_y (MPa)');
  set([tt xt yt], 'FontName', 'bookman', 'FontSize', 16);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 16);
  legend(p1,str_epdot,-1);

  %
  % Compute JC data for stress vs strain (at various temperatures)
  %
  [n, m] = size(T);
  for i=1:m
    epdot_3 = 1.0;
    for j=1:m
      sig_JC_3(i,j) = JC(epdot_3, ep(i), T(j));
      sig_SCG_3(i,j) = SCG(epdot_3, ep(i), T(j));
    end
  end
  fig3 = figure;
  set(fig3, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(ep, sig_JC_3(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(ep, sig_SCG_3(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
    set(p1(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    set(p2(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    str_T(j,:) = sprintf('T = %4s K',num2str(T(j)));
  end
  tt = title('Strain rate = 1.0/s, various T');
  xt = xlabel('\epsilon_p');
  yt = ylabel('\sigma_y (MPa)');
  set([tt xt yt], 'FontName', 'bookman', 'FontSize', 16);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 16);
  legend(p1,str_T,-1);

  %
  % Compute JC data for stress vs strain (at various strain rates)
  %
  [n, m] = size(T);
  for i=1:m
    T_4 = 300;
    for j=1:m
      sig_JC_4(i,j) = JC(epdot(j), ep(i), T_4);
      sig_SCG_4(i,j) = SCG(epdot(j), ep(i), T_4);
    end
  end
  fig4 = figure;
  set(fig4, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(ep, sig_JC_4(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(ep, sig_SCG_4(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
    set(p1(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    set(p2(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    str_epdot(j,:) = sprintf('d\\epsilon/dt = %5s/s',num2str(epdot(j),'%3.0e'));
  end
  tt = title('T = 300 K, various strain rates');
  xt = xlabel('\epsilon_p');
  yt = ylabel('\sigma_y (MPa)');
  set([tt xt yt], 'FontName', 'bookman', 'FontSize', 16);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 16);
  legend(p1,str_epdot,-1);

  %
  % Compute JC data for stress vs strain rate (at various temperatures)
  %
  [n, m] = size(T);
  for i=1:m
    ep_5 = 0.3;
    for j=1:m
      sig_JC_5(i,j) = JC(epdot(i), ep_5, T(j));
      sig_SCG_5(i,j) = SCG(epdot(i), ep_5, T(j));
    end
  end
  fig5 = figure;
  set(fig5, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(log10(epdot), sig_JC_5(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(log10(epdot), sig_SCG_5(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
    set(p1(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    set(p2(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    str_T(j,:) = sprintf('T = %4s K',num2str(T(j)));
  end
  tt = title('\epsilon_p = 0.3, various T');
  xt = xlabel('log (d\epsilon_p/dt) (/s)');
  yt = ylabel('\sigma_y (MPa)');
  set([tt xt yt], 'FontName', 'bookman', 'FontSize', 16);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 16);
  legend(p1,str_T,-1);
  set(gca, 'XLim', [-6 8]);

  %
  % Compute JC data for stress vs strain rate (at various strains)
  %
  [n, m] = size(T);
  for i=1:m
    T_6 = 300;
    for j=1:m
      sig_JC_6(i,j) = JC(epdot(i), ep(j), T_6);
      sig_SCG_6(i,j) = SCG(epdot(i), ep(j), T_6);
    end
  end
  fig6 = figure;
  set(fig6, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(log10(epdot), sig_JC_6(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(log10(epdot), sig_SCG_6(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
    set(p1(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    set(p2(j), 'Color', [1.0 - fac*j, 0.8*fac*j, 0.4*fac*j]);
    str_ep(j,:) = sprintf('\\epsilon = %4s',num2str(ep(j)));
  end
  tt = title('T = 300 K, various \epsilon_p');
  xt = xlabel('log (d\epsilon_p/dt) (/s)');
  yt = ylabel('\sigma_y (MPa)');
  set([tt xt yt], 'FontName', 'bookman', 'FontSize', 16);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 16);
  legend(p1,str_ep,-1);
  set(gca, 'XLim', [-6 8]);

%
%  Get JC yield stress
%
function [sigy] = JC(epdot, ep, T)

  A = 792.0e6;
  B = 510.0e6;
  C = 0.014;
  n = 0.26;
  m = 1.03;
  Tr = 294.0;
  Tm = 1793.0;

  eppart = A + B*ep^n;
  epdotpart = 1 + C*log(epdot);
  Tstar = (T - Tr)/(Tm - Tr);
  Tpart = 1 - Tstar^m;

  sigy = eppart*epdotpart*Tpart;


%
%  Get SCG yield stress
%
function [sigy] = SCG(epdot, ep, T)

  rho = 7850.0;
  rho0 = 7830.0;
  mu0 = 81.8e9;
  mu = calcmu(rho, rho0, T);

  beta = 2.0;
  ep0 = 0.0;
  sig0 = 1.15e9;
  n = 0.50;
  Ymax = 1.25e9;
  
  Ya = 1 + beta*(ep + ep0);
  siga = min(sig0*Ya^n, Ymax);

  sigt = calcYt(epdot, T);

  sigy = (sigt + siga)*mu/mu0;
  %[mu mu0 sigt siga sigy]

%
% Compute shear modulus
%
function [mu] = calcmu(rho, rho0, T)

  P = calcP(rho, rho0, T);
  Tm = calcTm(rho, rho0);

  mu0 = 81.8e9;
  dmu_dp_mu0 = 20.6e-12;
  dmu_dT_mu0 = 0.16e-3;

  That = T/Tm;
  if (That > 1.04)
    mu = 0;
  else
    eta = (rho/rho0)^(1/3);
    mu = mu0*(1 + dmu_dp_mu0*P/eta - dmu_dT_mu0*(T - 300));
  end

%
% Compute melting temperature
%
function [Tm] = calcTm(rho, rho0)

  Tm0 = 2310.0;
  Gamma0 = 3.0;
  a = 1.67;

  eta = rho/rho0;
  power = 2.0*(Gamma0 - a - 1.0/3.0);
  Tm = Tm0*exp(2.0*a*(1.0 - 1.0/eta))*eta^power;

%
% Compute pressure
%
function [p] = calcP(rho, rho0, T)

  C0 = 3574.0;
  Gamma0 = 1.69;
  S_alpha = 1.92;

  Cv = 477.0;
  zeta = rho/rho0 - 1;
  E = Cv*T;

  numer = rho0*C0^2*(1/zeta + 1 - 0.5*Gamma0);
  denom = 1/zeta + 1 - S_alpha;
  p = numer/denom^2 + Gamma0*E;

%
% Compute Yt
%
function [Yt] = calcYt(epdot, T)

  Uk = 0.31;
  k = 8.617385e-5;
  C1 = 3.1e6;
  C2 = 2.4e4;
  Yp = 7.0e8;

  tau_hi = Yp; 
  tau_lo = 1.0e-6; 
  tolerance = 1.0e-6;
  Yt = 0.5*(tau_hi+tau_lo);
  while ((tau_hi - tau_lo) > tolerance)

    % Compute f(tau_lo)
    A = 1 - tau_lo/Yp;
    B = 2*Uk/(k*T);
    C = B*A^2;
    D = 1/C1;
    E = D*exp(C);
    F = C2/tau_lo;
    G = E + F;
    f_lo = epdot - 1.0/G;

    % Compute f(tau)
    A = 1 - Yt/Yp;
    B = 2*Uk/(k*T);
    C = B*A^2;
    D = 1/C1;
    E = D*exp(C);
    F = C2/Yt;
    G = E + F;
    f = epdot - 1.0/G;

    % Check closeness
    if (f_lo*f > 0) 
      tau_lo = Yt;
    else
      tau_hi = Yt;
    end
    Yt = 0.5*(tau_hi + tau_lo);
  end
