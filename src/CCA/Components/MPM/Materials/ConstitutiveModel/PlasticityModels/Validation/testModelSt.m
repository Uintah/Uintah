function testModelSt

  %
  % Test the specific heat model
  %
  testSpecificHeatModel;

  %
  % Test the melting temperature model
  %
  testMeltTempModel;

  %
  % Test the EOS model
  %
  testEOSModel;

  %
  % Test the shear modulus models
  %
  testMuModel;

function testMuModel

  %
  % Experimental data from Fukuhara 1993
  %
  load ShearModVsTempS10C.dat;
  load ShearModVsTempSUS304.dat;
  muT1 = ShearModVsTempS10C;
  muT2 = ShearModVsTempSUS304;

  rho0 = 7830.0;
  eta = [0.9 1.0 1.1];
  [nr, mr] = size(eta);
  for i=1:mr
    rho(i) = eta(i)*rho0;
  end
  
  for i=1:1001
    T(i) = 20*(i-1);
  end

  [nt, mt] = size(T);
  for i=1:mr
    Tm(i) = calcTmBPS(rho(i), rho0);
    for j=1:mt
      P(i,j) = calcP(rho(i), rho0, T(j), T(j));
      muSCG(i,j) = calcmuSCG(rho(i), rho0, Tm(i), P(i,j), T(j));
      muNP(i,j) = calcmuNP(rho(i), rho0, Tm(i), P(i,j), T(j));
      muMTS(i,j) = calcmuMTS(T(j), Tm(i));
    end
  end

  %
  % Plot mu vs T/Tm for various rho (MTS model)
  %
  figure;
  Tm1 = calcTmBPS(rho0, rho0);
  p1 = plot(muT1(:,1)/Tm1, muT1(:,2), 'go', 'LineWidth', 3); hold on;
  p2 = plot(muT2(:,1)/Tm1, muT2(:,2), 'bs', 'LineWidth', 3); hold on;
  for i=1:mr
    p3(i) = plot(T/Tm(i), muMTS(i,:)/10^9, 'k-.', 'LineWidth', 3); hold on;
  end
  set(p3(1), 'LineStyle', '--', 'Color', [0.1 0.1 0.9]);
  set(p3(2), 'LineStyle', '-', 'Color', [0.9 0.1 0.1]);
  set(p3(3), 'LineStyle', '-.', 'Color', [0.4 0.7 0.1]);
  set(gca, 'XLim', [0 1.1], 'YLim', [0 120] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('T/T_m', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('Shear Modulus (GPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([p1 p2 p3(1) p3(2) p3(3)], ...
         'AISI 1010', 'SAE 304SS', ...
         'MTS (\eta = 0.9)','MTS (\eta = 1.0)', 'MTS (\eta = 1.1)');
  axis square;

  %
  % Plot mu vs T/Tm for various rho (SCG model)
  %
  figure;
  Tm1 = calcTmBPS(rho0, rho0);
  p1 = plot(muT1(:,1)/Tm1, muT1(:,2), 'go', 'LineWidth', 3); hold on;
  p2 = plot(muT2(:,1)/Tm1, muT2(:,2), 'bs', 'LineWidth', 3); hold on;
  for i=1:mr
    p3(i) = plot(T/Tm(i), muSCG(i,:)/10^9, 'k--', 'LineWidth', 3); hold on;
  end
  set(p3(1), 'LineStyle', '--', 'Color', [0.1 0.1 0.9]);
  set(p3(2), 'LineStyle', '-', 'Color', [0.9 0.1 0.1]);
  set(p3(3), 'LineStyle', '-.', 'Color', [0.4 0.7 0.1]);
  set(gca, 'XLim', [0 1.1], 'YLim', [0 120] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('T/T_m', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('Shear Modulus (GPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([p1 p2 p3(1) p3(2) p3(3)], ...
         'AISI 1010', 'SAE 304SS', ...
         'SCG (\eta = 0.9)','SCG (\eta = 1.0)', 'SCG (\eta = 1.1)');
  axis square;

  %
  % Plot mu vs T/Tm for various rho (MTS model)
  %
  figure;
  Tm1 = calcTmBPS(rho0, rho0);
  p1 = plot(muT1(:,1)/Tm1, muT1(:,2), 'go', 'LineWidth', 3); hold on;
  p2 = plot(muT2(:,1)/Tm1, muT2(:,2), 'bs', 'LineWidth', 3); hold on;
  for i=1:mr
    p3(i) = plot(T/Tm(i), muNP(i,:)/10^9, 'k-','LineWidth', 3); hold on;
  end
  set(p3(1), 'LineStyle', '--', 'Color', [0.1 0.1 0.9]);
  set(p3(2), 'LineStyle', '-', 'Color', [0.9 0.1 0.1]);
  set(p3(3), 'LineStyle', '-.', 'Color', [0.4 0.7 0.1]);
  set(gca, 'XLim', [0 1.1], 'YLim', [0 120] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('T/T_m', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('Shear Modulus (GPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([p1 p2 p3(1) p3(2) p3(3)], ...
         'AISI 1010', 'SAE 304SS', ...
         'NP (\eta = 0.9)','NP (\eta = 1.0)', 'NP (\eta = 1.1)');
  axis square;

function testEOSModel

  fig = figure;
  set(fig, 'Position', [379 482 711 562]);

  %
  % Load EOS data from Bancroft et al., JAP, 1956, 27(3), 291.
  %
  load EOSIronBancroft.dat
  invEtaEx0 = EOSIronBancroft(:,1);
  presEx0 = EOSIronBancroft(:,2)*1.0e11;
  etaEx0 = 1.0./invEtaEx0;
  pEx0 = plot(etaEx0, presEx0/10^9, 'o'); hold on;
  set(pEx0, 'LineWidth', 2, 'MarkerSize', 8);
  set(pEx0, 'Color', [0.9 0.0 0.0]);

  %
  % Load EOS data from Katz et al., JAP, 1959, 30(4), 568.
  %
  load EOSMildSteelKatz.dat
  invEtaEx1 = EOSMildSteelKatz(:,1);
  presEx1 = EOSMildSteelKatz(:,2)*10^8;
  etaEx1 = 1.0./invEtaEx1;
  pEx1 = plot(etaEx1, presEx1/10^9, 's'); hold on;
  set(pEx1, 'LineWidth', 2, 'MarkerSize', 8);
  set(pEx1, 'Color', [0.0 0.9 0.4]);
  
  %
  % Load EOS data from McQueen et al., 1970
  %
  load EOSIronMcQueenPV.dat
  invEtaEx2 = EOSIronMcQueenPV(:,1);
  presEx2 = EOSIronMcQueenPV(:,2)*1.0e11;
  etaEx2 = 1.0./invEtaEx2;
  pEx2 = plot(etaEx2, presEx2/1.0e9, 'p'); hold on;
  set(pEx2, 'LineWidth', 2, 'MarkerSize', 8);
  set(pEx2, 'Color', [0.75 0.25 1.0]);

  load EOSSteelMcQueenPV.dat
  rho0Ex3 = EOSSteelMcQueenPV(1,1);
  rhoEx3 = EOSSteelMcQueenPV(:,1);
  presEx3 = EOSSteelMcQueenPV(:,2)*1.0e11;
  etaEx3 = rhoEx3/rho0Ex3;
  pEx3 = plot(etaEx3, presEx3/1.0e9, '^'); hold on;
  set(pEx3, 'LineWidth', 2, 'MarkerSize', 8);
  set(pEx3, 'Color', [0.25 0.75 1.0]);

  %
  % Load EOS data from Barker et al., JAP, 1974, 45(11), 4872.
  %
  load EOSIronBarker.dat
  invEtaEx4 = EOSIronBarker(:,1);
  presEx4 = EOSIronBarker(:,2);
  etaEx4 = 1.0./invEtaEx4;
  pEx4 = plot(etaEx4, presEx4, 'd'); hold on;
  set(pEx4, 'LineWidth', 2, 'MarkerSize', 8);
  set(pEx4, 'Color', [0.01 0.01 0.9]);
  
  %
  % Load EOS data from Gust et al., 1979
  %
  load EOS4340GustPV.dat
  [m,n] = size(EOS4340GustPV);
  vol0Ex5 = EOS4340GustPV(m,1);
  volEx5 = EOS4340GustPV(:,1);
  presEx5 = EOS4340GustPV(:,2);
  etaEx5 = vol0Ex5./volEx5;
  pEx5 = plot(etaEx5, presEx5, 'v'); hold on;
  set(pEx5, 'LineWidth', 2, 'MarkerSize', 8);
  set(pEx5, 'Color', [0.0 0.0 0.0]);

  %
  % Plot P vs eta for various T 
  %
  rho0 = 7830.0;
  etaMin = 0.9;
  etaMax = 1.7;
  neta = 100;
  deta = (etaMax-etaMin)/neta;
  for i=1:neta+1
   eta(i) = etaMin + (i-1)*deta;
  end
  Tp = [300 1040 1800];
  for i=1:length(Tp)
    for j=1:length(eta)
      rho(j) = eta(j)*rho0;
      P(i,j) = calcP(rho(j), rho0, Tp(i), Tp(1));
    end
  end
  for i=1:length(Tp)
    p2(i) = plot(eta, P(i,:)/10^9, 'LineWidth', 2); hold on;
  end
  set(p2(1), 'Color', [0.0 0.0 1.0], 'LineStyle', '-');
  set(p2(2), 'Color', [1.0 0.0 0.0], 'LineStyle', '--');
  set(p2(3), 'Color', [0.21 0.79 0.13], 'LineStyle', '-.');
  set(gca, 'XLim', [0.9 1.7], 'XTick', [0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7]);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\eta = \rho/\rho_0', 'FontName', 'bookman', 'FontSize', 16);
  ylabel(' Pressure (GPa) ', 'FontName', 'bookman', 'FontSize', 16);

  legend([pEx0 pEx1 pEx2 pEx3 pEx4 pEx5 ...
          p2(1) p2(2) p2(3)], ...
         'Bancroft et al.(Iron)(1956)', 'Katz et al.(Steel)(1959)', ...
         'McQueen et al.(Iron)(1970)', 'McQueen et al.(304SS)(1970)',...
         'Barker et al.(Iron) (1974)','Gust et al.(4340)(1979)', ...
         'Model (300K)', 'Model (1040K)', 'Model (1800K)');
  axis square;

%
% Test the melting temperature model
%
function testMeltTempModel

  figure;

  load MeltTempIronBura1.dat
  load MeltTempIronBura2.dat
  load MeltTempIronBura3.dat
  Pex = MeltTempIronBura1(:,1);
  Tmex = MeltTempIronBura1(:,2);
  TmexLo = MeltTempIronBura2(:,2);
  TmexHi = MeltTempIronBura3(:,2);
  TmexU = TmexHi - Tmex;
  TmexL = Tmex - TmexLo;

  p0 = errorbar(Pex, Tmex, TmexL, TmexU, 'o'); hold on;
  set(p0, 'LineWidth', 2, 'Color', [0.1 0.5 0.6]);
  set(p0, 'MarkerSize', 8, 'MarkerFaceColor', [0.1 0.5 0.6]);

  rho0 = 7830.0;
  T = 300.0;
  T0 = T;
  etaMax = 1.9;
  etaMin = 0.8;
  nEta = 150.0;
  delEta = (etaMax-etaMin)/nEta;
  for i=1:nEta
    eta = etaMin + i*delEta;
    rho(i) = eta*rho0;
    P(i) = calcP(rho(i), rho0, T, T0);
    TmSCG(i) = calcTmSCG(rho(i), rho0);
    TmBPS(i) = calcTmBPS(rho(i), rho0);
  end
  
  p10 = plot(P*1.0e-9, TmSCG, '-.'); hold on;
  p11 = plot(P*1.0e-9, TmBPS, '-'); hold on;
  set(p10, 'LineWidth', 2, 'Color', [0.9 0.0 0.0]);
  set(p10, 'MarkerSize', 8, 'MarkerFaceColor', [0.9 0.0 0.0]);
  set(p11, 'LineWidth', 2, 'Color', [0.15 0.0 0.85]);
  set(p11, 'MarkerSize', 8, 'MarkerFaceColor', [0.15 0.0 0.85]);

  set(gca, 'XLim', [-50 350], 'XTick', [-50 0 50 100 150 200 250 300 350]);
  set(gca, 'YLim', [0 10000], 'YTick', [0 2000 4000 6000 8000 10000]);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel(' Pressure (GPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel(' T_m (K) ', 'FontName', 'bookman', 'FontSize', 16);
  axis square;
  legend([p0(2) p10 p11], ...
         'Burakovsky et al. (2000)','SCG Melt Model', ...
          'BPS Melt Model');

function testSpecificHeatModel

  figure;
  %
  % Calculate the Debye temperature
  %
  h = 6.626068e-34;
  kb = 1.3806503e-23;
  M = 55.845*1.66043998903379e-27;
  G = 81.8e9;
  rho = 7830.0;
  thetaD = 1/kb*(h/(2*pi)*(9*pi^2*rho/M)^(1/3)*(G/rho)^(1/2))

  %
  % Expt data Cp (From Wallace JAP 1960, 31, 168-176)
  %
  Tex00 = [ 25  50 100 150 200 250 300 350 400 ...
           450 500 550 600 650 700 730 750 760 ...
           765 769 770 780 800 850 900 950 1000 ...
           1050];
  Cpex00 = [0.104 0.108 0.114 0.120 0.125 0.130 0.135 0.139 0.144 ...
            0.150 0.157 0.164 0.174 0.189 0.213 0.238 0.291 0.321 ...
            0.385 0.213 0.199 0.187 0.173 0.166 0.143 0.145 0.147 ...
            0.150];
  Tex00 = Tex00 + 273;
  Cpex00 = Cpex00*4.184*1.0e3;
  p000 = plot(Tex00, Cpex00, 'd'); hold on;
  set(p000, 'LineWidth', 2, 'Color', [0.1 0.5 0.6]);

  Tc = 1040.0;
  for i=1:length(Tex00)
    tt(i) = Tex00(i)/Tc - 1;
  end
  fid = fopen('CpStExpt2.dat','w');
  for i=1:length(Tex00)
    fprintf(fid,'[%f,%f],',tt(i),Cpex00(i));
  end
  fclose(fid);

  %
  % Expt data Cp (From ASM metals handbook v1 1978 p.149)
  %
  Tex0 =  [75  175 275 375 475 575 675 775];
  Cpex0 = [477 515 544 595 657 737 825 833];
  Tex0 = Tex0 + 273;
  p00 = plot(Tex0, Cpex0, 's'); hold on;
  set(p00, 'LineWidth', 2, 'Color', [0.4 0.1 0.7]);

  %
  % Expt data Cp (From Shacklette, PRB v 9 n 9)
  %
  Tex = [1010.2 1012.5 1016.6 1018.5 1021.7 1023.5 1025.0 1026.7 1028.5 ...
        1030.0 1031.7 1033.0 1034.5 1036.5 1037.9 1038.4 1039.0 1039.6 ...
        1039.9 1040.2 1040.5 1040.8 1041.0 1041.2 1041.3 1041.4 1041.6 ...
        1041.7 1042.3 1042.9 1043.8 1045.3 1047.5 1050.7 1052.2 1055.4 ...
        1057.8 1060.2 1065.4 1070.7 1075.0 1079.7];
  Cpex = [52.553 53.187 54.151 54.788 55.804 56.354 56.861 57.587 58.308 ...
          59.079 59.890 60.704 61.516 63.283 64.663 65.179 65.958 67.127 ...
          67.648 68.517 69.256 70.384 71.517 72.474 69.504 64.181 61.691 ...
          60.337 57.017 55.483 53.869 52.535 50.986 49.474 49.067 48.340 ...
          47.926 47.511 46.810 46.415 46.203 46.029];
  Cpex = Cpex/55.847*1.0e3;
  p0 = plot(Tex, Cpex, 'o'); hold on;
  set(p0, 'LineWidth', 2, 'Color', [0.1 0.7 0.2]);

  fid = fopen('CpStExpt.dat','w');
  for i=1:length(Tex00)
    fprintf(fid,'[%f,%f],',Tex00(i),Cpex00(i));
  end
  for i=1:length(Tex0)
    fprintf(fid,'[%f,%f],',Tex0(i),Cpex0(i));
  end
  for i=1:length(Tex)
    fprintf(fid,'[%f,%f],',Tex(i),Cpex(i));
  end
  fclose(fid);

  %
  % Plot Cp vs T
  % 
  Tmin = 250.0;
  Tmax = 1800.0;
  nT = 100;
  delT = (Tmax-Tmin)/nT;
  for i=1:nT
    Tcp(i) = Tmin + i*delT;
    Cp(i)  = calcCpFit(Tcp(i));
  end
  p1 = plot(Tcp, Cp);
  set(p1, 'LineWidth', 2, 'Color', [1.0 0.25 0.75]);
  set(gca, 'XLim', [200 1800], 'XTick', [200 600 1000 1400 1800]);
  set(gca, 'YLim', [400 1800], 'YTick', [400 600 800 1000 1200 1400 1600 1800]);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel(' T (K)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel(' C_p (J/kg-K) ', 'FontName', 'bookman', 'FontSize', 16);
  axis square;
  legend([p000 p00 p0 p1], ...
         'Wallace et al. (1960)','ASM (1978)','Shacklette (1974)','Model');

function [p] = calcP(rho, rho0, T, T0)

  %
  % Best fit to pressure data but does not satisfy Us-Up curve
  %
  eta = rho/rho0;
  %C0 = 3574.0;
  %S_alpha = 1.92;
  %Gamma0 = 1.69;
  %Gamma0 = (1-eta)*0.5 + eta*1.89;

  % Data from Brown and Gust 79
  C0 = 3935.0;
  S_alpha = 1.578;
  Gamma0 = 1.69;

  %
  % Wilkins Los Alamos EOS data for steel
  %
  %C0 = 4600.0;
  %Gamma0 = 2.17;
  %S_alpha = 1.49;

  %Cv = 477.0;
  Cv = calcCpFit(T);
  zeta = rho/rho0 - 1;
  E = Cv*(T-T0)*rho0;

  if (rho == rho0)
    p = Gamma0*E;
  else
    numer = rho0*C0^2*(1/zeta + 1 - 0.5*Gamma0);
    denom = 1/zeta + 1 - S_alpha;
    p = numer/denom^2 + Gamma0*E;

    %
    % Gust version of Gruneisen EOS
    %
    %t0 = rho0*C0^2*zeta;
    %t1 = t0*((1-Gamma0/2) + 2*(S_alpha-1))*zeta;
    %t2 = t0*(2*(1-Gamma0/2)*(S_alpha-1) + 3*(S_alpha-1)^2)*zeta^2;
    %p = t0 + t1 + t2 + Gamma0*E;
  end

function [Cp] = calcCp(T)

  if (T < 1000.0) 
    Cp = 1.0e3*(0.1156 + 7.454e-4*T + 12404/T^2);
  else
    Cp = calcCpLSS(T);
  end

function [Cp] = calcCpLSS(T)

  M = 55.847;
  Tc = 1041.32;
  C = 100;
  Ap = 7.503;
  Bp = 22.06;
  alpham = -0.12;
  alphap = -0.12;
  Am = Ap/1.036;
  Bm = (Bp + 14.42)/1.036;
  t = T/Tc - 1;
  if (T < Tc) 
    Cp = Am*(1/abs(t)^alpham - 1)/alpham + Bm + C*t;
  else
    Cp = Ap*(1/t^alphap - 1)/alphap + Bp + C*t;
  end
  Cp = Cp/M*10^3;

function [Cp] = calcCpFit(T)

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

function [Tm] = calcTmSCG(rho, rho0)

  Tm0 = 1793.0;
  Gamma0 = 1.67;
  a = 1.67;

  eta = rho/rho0;
  power = 2.0*(Gamma0 - a - 1.0/3.0);
  Tm = Tm0*exp(2.0*a*(1.0 - 1.0/eta))*eta^power;

function [Tm] = calcTmBPS(rho, rho0)

  %
  % Constants and derivative from Guinan and Steinberg, 1974
  %
  B0 = 1.66e11; 
  dB_dp0 = 5.29;
  G0 = 0.819e11;
  dG_dp0 = 1.8;

  %
  % Calculate the pressure using Murnaghan EOS
  %
  eta = rho/rho0;
  %p = B0/dB_dp0*(eta^dB_dp0 - 1);
  p = calcP(rho, rho0, 300, 300);

  %
  % BPS parameters for Fe at T = 300K and p = 0
  %
  kappa = 1;  %Screw dislocation
  %kappa = 1.5;  %Edge dislocation
  %kappa = 1.25;  % Mixed
  %z = 12.0; % fcc-hcp lattice
  z = 8.0; % bcc lattice
  b2rhoTm = 0.64;
  %b2rhoTm = b2rhoTm-0.14;
  b2rhoTm = b2rhoTm+0.14;
  alpha = 2.9;
  %lambda = 1.41; % fcc/hcp lattice
  lambda = 1.30; % bcc lattice
  %lambda = 1.33;
  a = 5.4057*0.53e-10;
  %vws = a^3/4; % fcc crystal
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
  %Tm = Tm0*Bfac^(-1/dB_dp0)*(1.0 + Gfac*Bfac^(-1.0/(3.0*dB_dp0)));
  Tm = Tm0/eta*(1.0 + Gfac/eta^(1/3));

function [mu] = calcmuMTS(T, Tm)

  %
  % Compute mu
  %
  mu_0 = 85.0e9;
  D = 1.0e10;
  T_0 = 298;
  if (T == 0)
    T = 0.01;
  end
  That = T/Tm;
  if (That > 1.0)
    mu = 0;
  else
    mu = mu_0 - D/(exp(T_0/T) - 1);
  end

function [mu] = calcmuSCG(rho, rho0, Tm, P, T)

  mu0 = 81.9e9;
  dmu_dp = 1.8;
  dmu_dp_mu0 = dmu_dp/mu0;
  dmu_dT = 0.0387e9;
  dmu_dT_mu0 = dmu_dT/mu0;

  That = T/Tm;
  if (That > 1.0)
    mu = 0;
  else
    eta = (rho/rho0)^(1/3);
    mu = mu0*(1 + dmu_dp_mu0*P/eta - dmu_dT_mu0*(T - 300));
  end

function [mu] = calcmuNP(rho, rho0, Tm, P, T)

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

