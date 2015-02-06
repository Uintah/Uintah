function plotDataJC

  clear all; close all;
  %color = ['rgbmkrgbmkrgbmk'];
  %ep = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.5 1.7 2.0];
  %epdot = [1.0e-6 1.0e-5 1.0e-4 1.0e-3 1.0e-2 1.0e-1 1.0 1.0e1 1.0e2 1.0e3 1.0e4 ...
  %         1.0e5 1.0e6 1.0e7 1.0e8];
  %T = [300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700];
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
      sig_ZA_1(i,j) = ZA(epdot_1, ep(j), T(i));
    end
  end
  fig1 = figure;
  set(fig1, 'Position', [412 313 915 632]);
  fac = 1/m;
  for j=1:m
    p1(j) = plot(T, sig_JC_1(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(T, sig_ZA_1(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
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
      sig_ZA_2(i,j) = ZA(epdot(j), ep_2, T(i));
    end
  end
  fig2 = figure;
  set(fig2, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(T, sig_JC_2(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(T, sig_ZA_2(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
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
      sig_ZA_3(i,j) = ZA(epdot_3, ep(i), T(j));
    end
  end
  fig3 = figure;
  set(fig3, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(ep, sig_JC_3(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(ep, sig_ZA_3(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
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
      sig_ZA_4(i,j) = ZA(epdot(j), ep(i), T_4);
    end
  end
  fig4 = figure;
  set(fig4, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(ep, sig_JC_4(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(ep, sig_ZA_4(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
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
      sig_ZA_5(i,j) = ZA(epdot(i), ep_5, T(j));
    end
  end
  fig5 = figure;
  set(fig5, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(log10(epdot), sig_JC_5(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(log10(epdot), sig_ZA_5(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
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
      sig_ZA_6(i,j) = ZA(epdot(i), ep(j), T_6);
    end
  end
  fig6 = figure;
  set(fig6, 'Position', [412 313 915 632]);
  for j=1:m
    p1(j) = plot(log10(epdot), sig_JC_6(:,j)/10^6, 'k-', 'LineWidth',2); hold on;
    p2(j) = plot(log10(epdot), sig_ZA_6(:,j)/10^6, 'k--', 'LineWidth', 2); hold on;
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
%  Get ZA yield stress
%
function [sigy] = ZA(epdot, ep, T)

  %B = 770e6;
  %beta_0 = 8.6e-3;
  %beta_1 = 0.36e-3;
  %c_0 = 800e6;
  %K = 1100e6;
  %n = 0.58;
  B = 500e6;
  beta_0 = 5.0e-4;
  beta_1 = 2.0e-5;
  c_0 = 370e6;
  K = 400e6;
  n = 0.35;
  beta = beta_0 - beta_1*log(epdot);
  sigy = c_0 + B*exp(-beta*T) + K*ep^n;


