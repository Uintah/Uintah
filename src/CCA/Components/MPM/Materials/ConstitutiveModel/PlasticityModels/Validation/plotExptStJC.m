function plotExptSt

  clear all;

  fig = figure;
  set(fig, 'Position', [378 479 1147 537]);

  load ../Expt/St001Temp.dat;
  ASM = St001Temp;

  %
  % 0.001/s 298K 923K temper
  %
  E = 213.0e9;
  sigY1 = ASM(5,3);
  eps1 = 0.002 + sigY1/E;
  sigY2 = ASM(5,2);
  eps2 = 0.8*ASM(5,4)/100.0;
  AlASM298K1e_5(1,2) = sigY1*(1.0+eps1);
  AlASM298K1e_5(1,1) = log(1.0+eps1);
  AlASM298K1e_5(2,2) = sigY2*(1.0+eps2);
  AlASM298K1e_5(2,1) = log(1.0+eps2);
  epsEx = AlASM298K1e_5(:,1);
  seqEx = AlASM298K1e_5(:,2);
  pexp1e_5298 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp1e_5298,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0;
  epdot = 0.001;
  rhomax = 7831.0;
  T = 298.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  %
  % 500/s 298K
  %
  load ../Expt/St500298K.dat
  epsEx = St500298K(:,1)*1.0e-2;
  seqEx = St500298K(:,2);
  pexp500298 = plot(epsEx, seqEx, 'o', 'LineWidth', 2); hold on;
  set(pexp500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-5;
  epdot = 500.0;
  rhomax = 7850.0;
  T = 298.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %
  % 500/s 573K
  %
  load ../Expt/St500573K.dat
  epsEx = St500573K(:,1)*1.0e-2;
  seqEx = St500573K(:,2);
  pexp500573 = plot(epsEx, seqEx, 's', 'LineWidth', 2); hold on;
  set(pexp500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-5;
  epdot = 500.0;
  rhomax = 7850.0;
  T = 573.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  %
  % 500/s 773K
  %
  load ../Expt/St500773K.dat
  epsEx = St500773K(:,1)*1.0e-2;
  seqEx = St500773K(:,2);
  pexp500773 = plot(epsEx, seqEx, 'd', 'LineWidth', 2); hold on;
  set(pexp500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-5;
  epdot = 500.0;
  rhomax = 7850.0;
  T = 773.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  %fig = figure;
  %set(fig, 'Position', [378 479 1147 537]);

  %
  % 1500/s 298K
  %
  load ../Expt/St1500298K.dat
  epsEx = St1500298K(:,1)*1.0e-2;
  seqEx = St1500298K(:,2);
  pexp1500298 = plot(epsEx, seqEx, 'o', 'LineWidth', 2); hold on;
  set(pexp1500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-5;
  epdot = 1500.0;
  rhomax = 7850.0;
  T = 298.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %
  % 1500/s 573K
  %
  load ../Expt/St1500573K.dat
  epsEx = St1500573K(:,1)*1.0e-2;
  seqEx = St1500573K(:,2);
  pexp1500573 = plot(epsEx, seqEx, 's', 'LineWidth', 2); hold on;
  set(pexp1500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-5;
  epdot = 1500.0;
  rhomax = 7850.0;
  T = 573.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  %
  % 1500/s 973K
  %
  load ../Expt/St1500973K.dat
  epsEx = St1500973K(:,1)*1.0e-2;
  seqEx = St1500973K(:,2);
  pexp1500973 = plot(epsEx, seqEx, 'd', 'LineWidth', 2); hold on;
  set(pexp1500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-5;
  epdot = 1500.0;
  rhomax = 7850.0;
  T = 973.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  %
  % 1500/s 1173K
  %
  load ../Expt/St15001173K.dat
  epsEx = St15001173K(:,1)*1.0e-2;
  seqEx = St15001173K(:,2);
  pexp15001173 = plot(epsEx, seqEx, 'v', 'LineWidth', 2); hold on;
  set(pexp15001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-5;
  epdot = 1500.0;
  rhomax = 7850.0;
  T = 1173.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  %
  % 1500/s 1373K
  %
  load ../Expt/St15001373K.dat
  epsEx = St15001373K(:,1)*1.0e-2;
  seqEx = St15001373K(:,2);
  pexp15001373 = plot(epsEx, seqEx, 'p', 'LineWidth', 2); hold on;
  set(pexp15001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-5;
  epdot = 1500.0;
  rhomax = 7850.0;
  T = 1373.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  %fig = figure;
  %set(fig, 'Position', [378 479 1147 537]);

  %
  % 2500/s 773K
  %
  load ../Expt/St2500773K.dat
  epsEx = St2500773K(:,1)*1.0e-2;
  seqEx = St2500773K(:,2);
  pexp2500773 = plot(epsEx, seqEx, 'o', 'LineWidth', 2); hold on;
  set(pexp2500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-5;
  epdot = 2500.0;
  rhomax = 7850.0;
  T = 773.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  %
  % 2500/s 973K
  %
  load ../Expt/St2500973K.dat
  epsEx = St2500973K(:,1)*1.0e-2;
  seqEx = St2500973K(:,2);
  pexp2500973 = plot(epsEx, seqEx, 's', 'LineWidth', 2); hold on;
  set(pexp2500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-5;
  epdot = 2500.0;
  T = 973.0;
  rhomax = 7850.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  %
  % 2500/s 1173K
  %
  load ../Expt/St25001173K.dat
  epsEx = St25001173K(:,1)*1.0e-2;
  seqEx = St25001173K(:,2);
  pexp25001173 = plot(epsEx, seqEx, 'd', 'LineWidth', 2); hold on;
  set(pexp25001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-5;
  epdot = 2500.0;
  T = 1173.0;
  rhomax = 7850.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  %
  % 2500/s 1373K
  %
  load ../Expt/St25001373K.dat
  epsEx = St25001373K(:,1)*1.0e-2;
  seqEx = St25001373K(:,2);
  pexp25001373 = plot(epsEx, seqEx, 'v', 'LineWidth', 2); hold on;
  set(pexp25001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-5;
  epdot = 2500.0;
  T = 1373.0;
  rhomax = 7850.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax);
  [s2, e2] = adiJC(epdot, T, delT, rhomax);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);


%
% Isothermal JC data for stress vs strain 
%
function [sig, eps] = isoJC(epdot, T, delT, rho)

  epmax = 0.35;
  tmax = epmax/epdot;
  m = tmax/delT;
  ep = 0.0;
  for i=1:m
    sig(i) = JC(epdot, ep, T);
    eps(i) = ep;
    ep = ep + epdot*delT;
  end

%
% Adiabatic JC data for stress vs strain 
%
function [sig, eps] = adiJC(epdot, T0, delT, rho)

  %rho = 7830.0;
  Cp = 477.0;
  epmax = 0.35;
  fac = 0.9/(rho*Cp);
  tmax = epmax/epdot;
  m = tmax/delT;
  T = T0;
  ep = 0.0;
  for i=1:m
    sig(i) = JC(epdot, ep, T);
    eps(i) = ep;
    ep = ep + epdot*delT;
    T = T + sig(i)*epdot*fac*delT; 
  end

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
  ep0 = 1.0;
  epdot = epdot/ep0;
 
  eppart = A + B*ep^n;
  if (epdot < 1.0)
    epdotpart = (1 + epdot)^C; 
  else
    epdotpart = 1 + C*log(epdot);
  end
  Tstar = (T - Tr)/(Tm - Tr);
  Tpart = 1 - Tstar^m;

  sigy = eppart*epdotpart*Tpart;

