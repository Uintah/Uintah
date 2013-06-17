%
% Compare experimental stress-strain data with MTS model
%
function testMTSSt

  plotJCRc30Tension;
  plotJCRc30Shear;

  plotASMHRc32;

  plotLarsonRc38;
  plotLYRc38500;
  plotLYRc381500;
  plotLYRc382500;

  plotRc45Rc45_0001;
  plotRc45Rc45_1000;

  plotRc45Rc49_0001;
  plotRc45Rc49_1000;

%====================================================================
%
% Load experimental data from Johnson-Cook (Rc = 30)
%
function plotJCRc30Tension

  fig30 = figure;

  %
  % 0.002/s 298K
  %
  load SigEpsEp0002298KRc30Ten.dat;
  epEx = SigEpsEp0002298KRc30Ten(:,2);
  seqEx = SigEpsEp0002298KRc30Ten(:,3);
  pexp0002298 = plot(epEx, seqEx, '^-', 'LineWidth', 2); hold on;
  set(pexp0002298,'LineWidth',2,'MarkerSize',6,'Color',[0.2 0.5 1.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 30.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.5 1.0]);

  %
  % 570/s 298K
  %
  load SigEpsEp570298KRc30Ten.dat
  epEx = SigEpsEp570298KRc30Ten(:,2);
  seqEx = SigEpsEp570298KRc30Ten(:,3);
  pexp570298 = plot(epEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp570298,'LineWidth',2,'MarkerSize',6,'Color',[0.3 0.3 0.6]);

  delT = 1.0e-6;
  epdot = 570.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 30;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.3 0.3 0.6]);

  %
  % 604/s 500K
  %
  load SigEpsEp604500KRc30Ten.dat
  epEx = SigEpsEp604500KRc30Ten(:,2);
  seqEx = SigEpsEp604500KRc30Ten(:,3);
  pexp604500 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp604500,'LineWidth',2,'MarkerSize',6,'Color',[0.6 0.3 0.3]);

  delT = 1.0e-6;
  epdot = 604.0;
  T = 500.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 30;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.6 0.3 0.3]);

  %
  % 650/s 735K
  %
  load SigEpsEp650735KRc30Ten.dat
  epEx = SigEpsEp650735KRc30Ten(:,2);
  seqEx = SigEpsEp650735KRc30Ten(:,3);
  pexp650735 = plot(epEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp650735,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-6;
  epdot = 650.0;
  T = 735.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 30;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  set(gca, 'XLim', [0 0.8], 'YLim', [0 1800] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp0002298 pexp570298 pexp604500 pexp650735], ...
         '0.002/s 298 K JC(1985)', ...
         '570/s 298 K JC(1985)', ...
         '604/s 500 K JC(1985)', ...
         '650/s 735 K JC(1985)');
  axis square;

%====================================================================
function plotJCRc30Shear

  fig30 = figure;

  %
  % 0.009/s 298K
  %
  load SigEpsEp0009298KRc30Shear.dat;
  epEx = SigEpsEp0009298KRc30Shear(:,2);
  seqEx = SigEpsEp0009298KRc30Shear(:,3);
  pexp0009298 = plot(epEx, seqEx, 'v-', 'LineWidth', 3); hold on;
  set(pexp0009298,'LineWidth',3,'MarkerSize',6,'Color',[0.1 0.75 1.0]);

  delT = 1.0;
  epdot = 0.009;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 30.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.1 0.75 1.0]);

  %
  % 0.10/s 298K
  %
  load SigEpsEp010298KRc30Shear.dat;
  epEx = SigEpsEp010298KRc30Shear(:,2);
  seqEx = SigEpsEp010298KRc30Shear(:,3);
  pexp01298 = plot(epEx, seqEx, '<-', 'LineWidth', 3); hold on;
  set(pexp01298,'LineWidth',3,'MarkerSize',6,'Color',[0.2 0.8 0.2]);

  delT = 0.1;
  epdot = 0.1;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.8 0.2]);

  %
  % 1.1/s 298K
  %
  load SigEpsEp1_1298KRc30Shear.dat;
  epEx = SigEpsEp1_1298KRc30Shear(:,2);
  seqEx = SigEpsEp1_1298KRc30Shear(:,3);
  pexp1298 = plot(epEx, seqEx, '>-', 'LineWidth', 3); hold on;
  set(pexp1298,'LineWidth',3,'MarkerSize',6,'Color',[0.8 0.4 0.1]);

  delT = 0.01;
  epdot = 1.1;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.4 0.1]);

  set(gca, 'XLim', [0 1.0], 'YLim', [0 1800] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp0009298 pexp01298 pexp1298], ...
         '0.009/s 298 K (Shear) JC(1985)', ...
         '0.1/s 298 K (Shear) JC(1995)', ...
         '1.1/s 298 K (Shear) JC(1995)');
  axis square;

%====================================================================
%
% Load experimental data from Aerospace Structural Metals handbook
%
function plotASMHRc32

  fig40 = figure;

  %
  % 0.002/s 298K Rc 32
  %
  load SigEpsEp0002298KRc32.dat;
  epEx = SigEpsEp0002298KRc32(:,2);
  seqEx = SigEpsEp0002298KRc32(:,3);
  pexp0002298 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  Rc = 32.0;
  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %
  % 0.002/s 422K Rc 32
  %
  load SigEpsEp0002422KRc32.dat;
  epEx = SigEpsEp0002422KRc32(:,2);
  seqEx = SigEpsEp0002422KRc32(:,3);
  pexp0002422 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002422,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0;
  epdot = 0.002;
  T = 422.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  %
  % 0.002/s 589K Rc 32
  %
  load SigEpsEp0002589KRc32.dat;
  epEx = SigEpsEp0002589KRc32(:,2);
  seqEx = SigEpsEp0002589KRc32(:,3);
  pexp0002589 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002589,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 589.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 32.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  %
  % 0.002/s 644K Rc 32
  %
  load SigEpsEp0002644KRc32.dat;
  epEx = SigEpsEp0002644KRc32(:,2);
  seqEx = SigEpsEp0002644KRc32(:,3);
  pexp0002644 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002644,'LineWidth',2,'MarkerSize',6,'Color',[0.2 0.6 0.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 644.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 32;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.6 0.0]);

  set(gca, 'XLim', [0 0.2], 'YLim', [0 1200] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp0002298 pexp0002422 pexp0002589 pexp0002644], ...
         '0.002/s 298 K ASMH (1995)', ...
         '0.002/s 422 K ASMH (1995)', ...
         '0.002/s 589 K ASMH (1995)', ...
         '0.002/s 644 K ASMH (1995)');
  axis square;
         
%====================================================================
%
% Load experimental data from Larson (Rc = 38)
%
function plotLarsonRc38

  fig20 = figure;

  %
  % 0.0002/s 258 K
  %
  load SigEpsEp0002258KRc38.dat;
  epEx = SigEpsEp0002258KRc38(:,2);
  seqEx = SigEpsEp0002258KRc38(:,3);
  pexp00002258 = plot(epEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp00002258,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0002;
  T = 258.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 38.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %
  % 0.0002/s 298 K
  %
  load SigEpsEp0002298KRc38.dat;
  epEx = SigEpsEp0002298KRc38(:,2);
  seqEx = SigEpsEp0002298KRc38(:,3);
  pexp00002298 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp00002298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 1.0 0.2]);

  delT = 10.0;
  epdot = 0.0002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 38.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.2]);

  %
  % 0.0002/s 373 K
  %
  load SigEpsEp0002373KRc38.dat;
  epEx = SigEpsEp0002373KRc38(:,2);
  seqEx = SigEpsEp0002373KRc38(:,3);
  pexp00002373 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp00002373,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.1 0.1]);

  delT = 10.0;
  epdot = 0.0002;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 38.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.1 0.1]);

  set(gca, 'XLim', [0 0.8], 'YLim', [0 2000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp00002258 pexp00002298 pexp00002373], ...
         '0.0002/s 258 K Larson(1961)', ...
         '0.0002/s 298 K Larson(1961)', ...
         '0.0002/s 373 K Larson(1961)');
  axis square;

%====================================================================
function plotLYRc38500

  fig50 = figure;
  
  %
  % 500/s 298K
  %
  load SigEpsEp500298KRc38.dat
  epEx = SigEpsEp500298KRc38(:,2);
  seqEx = SigEpsEp500298KRc38(:,3);
  pexp500298 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %
  % 500/s 573K
  %
  load SigEpsEp500573KRc38.dat
  epEx = SigEpsEp500573KRc38(:,2);
  seqEx = SigEpsEp500573KRc38(:,3);
  pexp500573 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  %
  % 500/s 773K
  %
  load SigEpsEp500773KRc38.dat
  epEx = SigEpsEp500773KRc38(:,2);
  seqEx = SigEpsEp500773KRc38(:,3);
  pexp500773 = plot(epEx, seqEx, '^-', 'LineWidth', 2); hold on;
  set(pexp500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 0.5]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 0.5]);

  set(gca, 'XLim', [0 0.12], 'YLim', [0 1600] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp500298 pexp500573 pexp500773], ...
         '500/s 298 K LY(1997)', ...
         '500/s 573 K LY(1997)', ...
         '500/s 773 K LY(1997)');
  axis square;
         
  %====================================================================


function plotLYRc381500

  E = 213.0e9;
  fig60 = figure;
  %set(fig3, 'Position', [378 479 1147 537]);
  %
  % 1500/s 298K
  %
  load SigEpsEp1500298KRc38.dat
  epEx = SigEpsEp1500298KRc38(:,2);
  seqEx = SigEpsEp1500298KRc38(:,3);
  pexp1500298 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp1500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %
  % 1500/s 573K
  %
  load SigEpsEp1500573KRc38.dat
  epEx = SigEpsEp1500573KRc38(:,2);
  seqEx = SigEpsEp1500573KRc38(:,3);
  pexp1500573 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp1500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  %
  % 1500/s 973K
  %
  load SigEpsEp1500973KRc38.dat
  epEx = SigEpsEp1500973KRc38(:,2);
  seqEx = SigEpsEp1500973KRc38(:,3);
  pexp1500973 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp1500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  %
  % 1500/s 1173K
  %
  load SigEpsEp15001173KRc38.dat
  epEx = SigEpsEp15001173KRc38(:,2);
  seqEx = SigEpsEp15001173KRc38(:,3);
  pexp15001173 = plot(epEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp15001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  %
  % 1500/s 1373K
  %
  load SigEpsEp15001373KRc38.dat
  epEx = SigEpsEp15001373KRc38(:,2);
  seqEx = SigEpsEp15001373KRc38(:,3);
  pexp15001373 = plot(epEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp15001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  set(gca, 'XLim', [0 0.24], 'YLim', [0 1800] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp1500298 pexp1500573 pexp1500973 pexp15001173 pexp15001373], ...
         '1500/s 298 K LY(1997)', ...
         '1500/s 573 K LY(1997)', ...
         '1500/s 873 K LY(1997)', ...
         '1500/s 1173 K LY(1997)', ...
         '1500/s 1373 K LY(1997)');
  axis square
         
  %====================================================================
         
function plotLYRc382500

  E = 213.0e9;
  fig70 = figure;
  %set(fig4, 'Position', [378 479 1147 537]);
  %
  % 2500/s 773K
  %
  load SigEpsEp2500773KRc38.dat
  epEx = SigEpsEp2500773KRc38(:,2);
  seqEx = SigEpsEp2500773KRc38(:,3);
  pexp2500773 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp2500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  %
  % 2500/s 973K
  %
  load SigEpsEp2500973KRc38.dat
  epEx = SigEpsEp2500973KRc38(:,2);
  seqEx = SigEpsEp2500973KRc38(:,3);
  pexp2500973 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp2500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  %
  % 2500/s 1173K
  %
  load SigEpsEp25001173KRc38.dat
  epEx = SigEpsEp25001173KRc38(:,2);
  seqEx = SigEpsEp25001173KRc38(:,3);
  pexp25001173 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp25001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  %
  % 2500/s 1373K
  %
  load SigEpsEp25001373KRc38.dat
  epEx = SigEpsEp25001373KRc38(:,2);
  seqEx = SigEpsEp25001373KRc38(:,3);
  pexp25001373 = plot(epEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp25001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  set(gca, 'XLim', [0 0.35], 'YLim', [0 1000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp2500773 pexp2500973 pexp25001173 pexp25001373], ...
         '2500/s 773 K LY(1997)', ...
         '2500/s 973 K LY(1997)', ...
         '2500/s 1173 K LY(1997)', ...
         '2500/s 1373 K LY(1997)');
  axis square;

  %====================================================================
         
function plotRc45Rc45_0001

  E = 213.0e9;
  fig01 = figure;

  %
  % Plot experimental data for 4340 steel Rc 45 (Rc45 et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load SigEpsEp0001173KRc45.dat
  epEx = SigEpsEp0001173KRc45(:,2);
  seqEx = SigEpsEp0001173KRc45(:,3);
  pexp00001173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp00001173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 45.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  load SigEpsEp0001298KRc45.dat
  epEx = SigEpsEp0001298KRc45(:,2);
  seqEx = SigEpsEp0001298KRc45(:,3);
  pexp00001298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp00001298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 45.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  load SigEpsEp0001373KRc45.dat
  epEx = SigEpsEp0001373KRc45(:,2);
  seqEx = SigEpsEp0001373KRc45(:,3);
  pexp00001373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp00001373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 45.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  set(gca, 'XLim', [0 0.3], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp00001173 pexp00001298 pexp00001373], ...
         '0.0001/s 173 K Chi(1989)', ...
         '0.0001/s 298 K Chi(1989)', ...
         '0.0001/s 373 K Chi(1989)');
  axis square;

  %====================================================================


function plotRc45Rc45_1000

  E = 213.0e9;
  fig11 = figure;

  %
  % Plot experimental data for 4340 steel Rc 45 (Rc45 et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load SigEpsEp1000173KRc45.dat
  epEx = SigEpsEp1000173KRc45(:,2);
  seqEx = SigEpsEp1000173KRc45(:,3);
  pexp1000173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp1000173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 45.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  load SigEpsEp1000298KRc45.dat
  epEx = SigEpsEp1000298KRc45(:,2);
  seqEx = SigEpsEp1000298KRc45(:,3);
  pexp1000298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp1000298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 45.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  load SigEpsEp1000373KRc45.dat
  epEx = SigEpsEp1000373KRc45(:,2);
  seqEx = SigEpsEp1000373KRc45(:,3);
  pexp1000373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp1000373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 45.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  set(gca, 'XLim', [0 0.1], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp1000173 pexp1000298 pexp1000373], ...
         '1000/s 173 K Chi(1989)', ...
         '1000/s 298 K Chi(1989)', ...
         '1000/s 373 K Chi(1989)');
  axis square;

  %====================================================================

function plotRc45Rc49_0001

  E = 213.0e9;
  fig00 = figure;

  %
  % Plot experimental data for 4340 steel Rc 49 (Rc49 et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load SigEpsEp0001173KRc49.dat
  epEx = SigEpsEp0001173KRc49(:,2);
  seqEx = SigEpsEp0001173KRc49(:,3);
  pexp00001173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp00001173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 49.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  load SigEpsEp0001298KRc49.dat
  epEx = SigEpsEp0001298KRc49(:,2);
  seqEx = SigEpsEp0001298KRc49(:,3);
  pexp00001298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp00001298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 49.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  load SigEpsEp0001373KRc49.dat
  epEx = SigEpsEp0001373KRc49(:,2);
  seqEx = SigEpsEp0001373KRc49(:,3);
  pexp00001373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp00001373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 49.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  set(gca, 'XLim', [0 0.25], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp00001173 pexp00001298 pexp00001373], ...
         '0.0001/s 173 K Chi(1989)', ...
         '0.0001/s 298 K Chi(1989)', ...
         '0.0001/s 373 K Chi(1989)');
  axis square;

  %====================================================================

function plotRc45Rc49_1000

  E = 213.0e9;
  fig10 = figure;

  %
  % Plot experimental data for 4340 steel Rc 49 (Rc49 et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load SigEpsEp1000173KRc49.dat
  epEx = SigEpsEp1000173KRc49(:,2);
  seqEx = SigEpsEp1000173KRc49(:,3);
  pexp1000173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp1000173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 49.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  load SigEpsEp1000298KRc49.dat
  epEx = SigEpsEp1000298KRc49(:,2);
  seqEx = SigEpsEp1000298KRc49(:,3);
  pexp1000298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp1000298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 49.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  load SigEpsEp1000373KRc49.dat
  epEx = SigEpsEp1000373KRc49(:,2);
  seqEx = SigEpsEp1000373KRc49(:,3);
  pexp1000373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp1000373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 49.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  set(gca, 'XLim', [0 0.1], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp1000173 pexp1000298 pexp1000373], ...
         '1000/s 173 K Chi(1989)', ...
         '1000/s 298 K Chi(1989)', ...
         '1000/s 373 K Chi(1989)');
  axis square;

%====================================================================
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

%
% Isothermal MTS data for stress vs strain 
%
function [sig, eps] = isoMTS(epdot, T0, delT, rhomax, epmax, Rc)

  rho0 = 7830.0;
  tmax = epmax/epdot;
  m = tmax/delT;
  ep = 0.0;
  delrho = (rhomax - rho0)/m;
  rho = rho0+0.1;
  T = T0;
  for i=1:m
    sig(i) = MTS(epdot, ep, T, T0, rho, rho0, Rc);
    eps(i) = ep;
    ep = ep + epdot*delT;
    rho = rho + delrho;
  end

%
% Adiabatic MTS data for stress vs strain 
%
function [sig, eps] = adiMTS(epdot, T0, delT, rhomax, epmax, Rc)

  rho0 = 7830.0;
  tmax = epmax/epdot;
  m = tmax/delT;
  delrho = (rhomax - rho0)/m;
  rho = rho0+0.1;
  T = T0;
  ep = 0.0;
  for i=1:m
    sig(i) = MTS(epdot, ep, T, T0, rho, rho0,  Rc);
    eps(i) = ep;
    ep = ep + epdot*delT;
    Cp = calcCp(T);
    fac = 0.9/(rho*Cp);
    T = T + sig(i)*epdot*fac*delT; 
    rho = rho + delrho;
  end

%
% Get MTS yield stress
%
function [sigy] = MTS(epdot, ep, T, T0, rho, rho0, Rc)

  %
  % Compute mu_0 (value of mu at T = 0)
  %
  P = calcP(rho, rho0, 0, 0);
  Tm = calcTm(rho, rho0);
  mu_0 = calcmu(rho, rho0, Tm, P, 0);

  %
  % Compute mu
  %
  P = calcP(rho, rho0, T, T0);
  Tm = calcTm(rho, rho0);
  mu = calcmu(rho, rho0, Tm, P, T);

  %
  % Get sigma_es0 and g_0es
  %
  [sigma_es0, g_0es] = getSigma_es0(Rc, T);

  %
  % Compute sigma_es
  %
  edot_0es = 1.0e7;
  kappa = 1.38e-23;
  b = 2.48e-10;
  sigma_es = sigma_es0*(epdot/edot_0es)^(kappa*T/(mu*b^3*g_0es)); 

  %
  % Get theta_0 and theta_IV
  %
  [theta_0, theta_IV] = getTheta0(Rc, epdot, T); 

  %
  % Compute sigma_e
  %
  sigma_e = computeSigma_e(theta_0, theta_IV, sigma_es, ep);

  %
  % Compute s_e
  %
  g_0e = 1.6;
  epdot_0e = 1.0e7;
  p_e = 2.0/3.0;
  q_e = 1.0;
  s_e = (1.0 - (kappa*T/(mu*b^3*g_0e)*log(epdot_0e/epdot))^(1/q_e))^(1/p_e);

  %
  % Get sigma_i and g_0i
  %
  [sigma_i, g_0i] = getSigmai(Rc, T);

  %
  % Compute s_i
  %
  epdot_0i = 1.0e8;
  p_i = 2.0/3.0;
  q_i = 1.0;
  s_i = (1.0 - (kappa*T/(mu*b^3*g_0i)*log(epdot_0i/epdot))^(1/q_i))^(1/p_i);

  %
  % Compute sigma/mu
  %
  sigma_a = 50.0e6;

  %
  % Compute sigma_y/mu
  %
  sigma_mu = sigma_a/mu + s_i*sigma_i/mu_0 + s_e*sigma_e/mu_0;

  %
  % Compute sigy
  %
  sigy = sigma_mu*mu;
  %[sigy sigma_a s_i sigma_i s_e sigma_e mu mu_0]

%
% Integrate dsigma_e/dep
%
function [sigma_e] = computeSigma_e(theta_0, theta_IV, sigma_es, ep)

  if (ep == 0)
    sigma_e = 0.0;
    return;
  end

  alpha = 3.0;
  sigma_e = 0;
  dep = ep/100;
  for i=1:101
    FX = tanh(alpha*sigma_e/sigma_es)/tanh(alpha);
    sigma_e = sigma_e + dep*(theta_0*(1.0-FX) + theta_IV*FX);
    %if (sigma_e > sigma_es)
    %  sigma_e = sigma_es;
    %  break;
    %end
  end

%
% Hardness vs yield stress 
%
function calcSigY(Tt,Rc)

  % Hardness (from ASMH)
  T = [205 315 425 540 650 705];
  sigy = [1860 1620 1365 1160 860 740];
  sigu = [1980 1760 1500 1240 1020 860];
  plot(T, sigy, 'r', T, sigu, 'b');
  HRC = [53 49.5 46 39 31 24];
  plot(HRC, sigy, 'r', HRC, sigu, 'b'); hold on;

  % Hardness (from other stress-strain data)
  Rc = [30 38 45 49 55]
  sy = [792 950 1268 1459 1756]
  p = polyfit(Rc, log(sy), 1)
  for i=1:100
    rrc(i) = 22 + (55-22)/100*i;
    ss(i) = p(1)*rrc(i) + p(2);
  end
  plot(Rc, sy,'go-')
  plot(rrc, exp(ss), 'm-')

%
% Calculate intersection with stress-strain curve at various strains
%
function [p] = intersectPoly(stress, strain, eps)

  %
  % Create vertical line
  %
  p1 = [eps 0.0];
  p2 = [eps 2500.0];

  %
  % Find intersection with polyline
  %
  p = p1;
  [n,m] = size(stress);
  for i=1:n-1
    p3(1) = strain(i,:);
    p3(2) = stress(i,:);
    p4(1) = strain(i+1,:);
    p4(2) = stress(i+1,:);
    [p, t1, t2] = intersect(p1,p2,p3,p4);
    if ((t2 >= 0.0) & (t2 <= 1.0))
      break;
    end
  end

%
% Find the intersection of two lines
%
function [p, t1, t2] = intersect(p1,p2,p3,p4)

  x1 = p1(1); x2 = p2(1); x3 = p3(1); x4 = p4(1);
  y1 = p1(2); y2 = p2(2); y3 = p3(2); y4 = p4(2);

  dx1 = x2 - x1;
  dy1 = y2 - y1;
  dx2 = x4 - x3;
  dy2 = y4 - y3;
  dx3 = x1 - x3;
  dy3 = y1 - y3;

  denom = dy2*dx1 - dx2*dy1;
  t1 = (dx2*dy3 - dy2*dx3)/denom;
  t2 = (dx1*dy3 - dy1*dx3)/denom;
  p(1) = x1 + t1*dx1;
  p(2) = y1 + t1*dy1;

%
% Calc theta_0 and theta_IV
%
function [theta_0, theta_IV] = getTheta0(Rc, epdot, T)

  %a_0 = 5102.4e6;
  %a_1 = 0.0;
  %a_2 = 0.0;
  %a_3 = 2.0758e6;

  %
  % Multiple temperature fit (true Sigeps)
  %
%  if (T > 1040)
%    a_0 = 21057.3;
%    a_1 = 4.4511e-13;
%    a_2 = -179.295;
%    a_3 = -7.26015;
%    aIV_0 = 10229.6;
%    aIV_1 = -3.33833e-13;
%    aIV_2 = -65.5715;
%    aIV_3 = -4.61843;
%  else
%    if (Rc == 30)
%      a_0 = -4618.45;
%      a_1 = -63.3439;
%      a_2 = 233.174;
%      a_3 = 35.2382;
%      aIV_0 = 1196.3;
%      aIV_1 = -24.0692;
%      aIV_2 = 94.7978;
%      aIV_3 = -2.01891;
%    elseif (Rc == 38)
%      a_0 = 11233.2;
%      a_1 = 50.9479;
%      a_2 = -51.7189;
%      a_3 = -3.4243;
%      aIV_0 = 4135.33;
%      aIV_1 = 16.427;
%      aIV_2 = 6.47717;
%      aIV_3 = -1.55081;
%    elseif (Rc == 45)
%      a_0 = 36731.6;
%      a_1 = 1.03449e-13;
%      a_2 = 58.3492;
%      a_3 = -84.1363;
%      aIV_0 = -2467.14;
%      aIV_1 = 1.41067e-14;
%      aIV_2 = -56.4667;
%      aIV_3 = 19.3153;
%    elseif (Rc == 49)
%      a_0 = -1523.72;
%      a_1 = 3.7618e-14;
%      a_2 = -220.38;
%      a_3 = 80.0986;
%      aIV_0 = 1896.52;
%      aIV_1 = -1.8809e-14;
%      aIV_2 = 13.634;
%      aIV_3 = 5.61732;
%    else
%      %a_0 = 10455.6;
%      %a_1 = -3.09901;
%      %a_2 = 4.85613;
%      %a_3 = 6.94408;
%      %aIV_0 = 1190.25;
%      %aIV_1 = -1.91056;
%      %aIV_2 = 14.6106;
%      %aIV_3 = 5.34072;
%      a_0 =  82395.4*Rc^3 + 75229.1*Rc^2 +   68398*Rc + 61895.6;
%      a_1 = -371.861*Rc^3 - 350.431*Rc^2 - 329.695*Rc - 309.643;
%      a_2 =  1399.09*Rc^3 + 1313.83*Rc^2 + 1231.74*Rc + 1152.77;
%      a_3 = -327.279*Rc^3 - 297.923*Rc^2 - 269.904*Rc - 243.196;
%      aIV_0 = -24950.4*Rc^3 - 22981.9*Rc^2 - 21091.1*Rc - 19276.7;
%      aIV_1 = -129.875*Rc^3 -  122.55*Rc^2 - 115.461*Rc - 108.603;
%      aIV_2 =  3.27627*Rc^3 +  12.345*Rc^2 + 20.8735*Rc + 28.8731;
%      aIV_3 =   63.524*Rc^3 + 58.4248*Rc^2 + 53.5405*Rc +  48.867;
%    end
%  end
%  theta_0 = a_0 + a_1*log(epdot) + a_2*sqrt(epdot) + a_3*T;
%  theta_0 = theta_0*1.0e6;
%  theta_IV = aIV_0 + aIV_1*log(epdot) + aIV_2*sqrt(epdot) + aIV_3*T;
%  theta_IV = theta_IV*1.0e6;

  %
  % Multiple temperature fit (true Sigeps)
  %
  if (T > 1040)
    a_0 = 18796.9;
    a_1 = -2.22555e-13;
    a_2 = -173.046;
    a_3 = -6.35958;
    aIV_0 = 3548.44;
    aIV_1 =  -2.22555e-13;
    aIV_2 = -37.1949;
    aIV_3 =  -1.03591;
  else
    if (Rc == 30)
      a_0 = -5210.11;
      a_1 = -62.2241;
      a_2 = 221.844;
      a_3 = 36.3249;
      aIV_0 = 281.672;
      aIV_1 =  -30.1434;
      aIV_2 = 62.5845;
      aIV_3 =  -0.443201;
    elseif (Rc == 38)
      a_0 = 10589.5;
      a_1 = 48.6378;
      a_2 = -60.4827;
      a_3 = -3.60117;
      aIV_0 = 1574.65;
      aIV_1 =  2.1881;
      aIV_2 = 4.83983;
      aIV_3 =  -1.05692;
    elseif (Rc == 45)
      a_0 = 40626.9;
      a_1 = -1.38716e-13;
      a_2 = 197.204;
      a_3 = -108.31;
      aIV_0 = -190.528;
      aIV_1 =  -7.64115e-15;
      aIV_2 = -5.26127;
      aIV_3 =  4.45097;
    elseif (Rc == 49)
      a_0 = -3517.65;
      a_1 = -3.7618e-14;
      a_2 = -221.699;
      a_3 = 79.7834;
      aIV_0 = -782.242;
      aIV_1 =  -7.05337e-15;
      aIV_2 = 30.2209;
      aIV_3 =  6.83384;
    else
      a_0 = 10622.2;
      a_1 = -3.39658;
      a_2 = 34.2165;
      a_3 = 1.0493;
      aIV_0 = 220.888;
      aIV_1 =  -6.98881;
      aIV_2 = 23.096;
      aIV_3 =  2.44617;
      %a_0 =  100853*Rc^3 + 92198.5*Rc^2 + 83942.5*Rc + 76077.4;
      %aIV_0 =  -4855.38*Rc^3 + -4489.91*Rc^2 + -4136.96*Rc + -3796.37;
      %a_1 =  -359.795*Rc^3 + -339.138*Rc^2 + -319.148*Rc + -299.816;
      %aIV_1 =  -91.0227*Rc^3 + -86.9817*Rc^2 + -83.0551*Rc + -79.2413;
      %a_2 =  1920.02*Rc^3 + 1793.22*Rc^2 + 1671.36*Rc + 1554.35;
      %aIV_2 =  85.8913*Rc^3 + 85.3999*Rc^2 + 84.8211*Rc + 84.1578;
      %a_3 =  -412.913*Rc^3 + -376.768*Rc^2 + -342.25*Rc + -309.328;
      %aIV_3 =  8.8103*Rc^3 + 8.12586*Rc^2 + 7.46698*Rc + 6.83329;
    end
  end
  theta_0 = a_0 + a_1*log(epdot) + a_2*sqrt(epdot) + a_3*T;
  theta_0 = theta_0*1.0e6;
  theta_IV = aIV_0 + aIV_1*log(epdot) + aIV_2*sqrt(epdot) + aIV_3*T;
  theta_IV = theta_IV*1.0e6;

%
% Get sigma_es0 and g_0es
%
function [sigma_es0, g_0es] = getSigma_es0(Rc, T)

  %
  % HY-100 fit
  %
  %sigma_es0 = 790e6;
  %g_0es = 0.112;

  %
  % Multiple temperature fit (true sig-eps)
  %
%  if (T > 1040)
%    g_0es = 0.20093;
%    sigma_es0 = 1.9473e3;
%  else
%    if (Rc == 30)
%      g_0es = 0.31659;
%      sigma_es0 = 6.8658e2;
%    elseif (Rc == 38)
%      g_0es = 0.2725;
%      sigma_es0 = 1.6926e3;
%    elseif (Rc == 45)
%      g_0es = 0.076556;
%      sigma_es0 = 1.6959e3;
%    elseif (Rc == 49)
%      g_0es = 0.62284;
%      sigma_es0 = 1.1429e3;
%    else
%      g_0es = 0.3409;
%      sigma_es0 = 488.67;
%      %g_0es = 8.6627e-04*Rc^3 - 9.9388e-02*Rc^2 + 3.7348*Rc - 45.667;
%      %sigma_es0 = -0.22428*Rc^3 + 16.992*Rc^2 - 248.32*Rc - 1101;
%    end
%  end
%  sigma_es0 = sigma_es0*1.0e6;

  %
  % Multiple temperature fit (sig-eps)
  %
  if (T > 1040)
    g_0es = 0.40382;
    sigma_es0 = 377.62;
  else
    if (Rc == 30)
      g_0es = 0.097526;
      sigma_es0 = 1147.9;
    elseif (Rc == 38)
      g_0es = 0.16867;
      sigma_es0 = 1118.0;
    elseif (Rc == 45)
      g_0es = 0.19275;
      sigma_es0 = 186.76;
    elseif (Rc == 49)
      g_0es = 0.74925;
      sigma_es0 = 283.44;
    else
      g_0es = 0.18071;
      sigma_es0 = 684.04;
      %g_0es = 6.6837e-04*Rc^3 - 7.5889e-02*Rc^2 + 2.8408*Rc - 34.871;
      %sigma_es0 = 1.2059*Rc^3 - 144.88*Rc^2 + 5647.0*Rc - 70427;
    end
  end
  sigma_es0 = sigma_es0*1.0e6;

%
% Get sigma_i and g_0i
%
function [sigma_i, g_0i] = getSigmai(Rc, T)

  %
  % Multiple temperature fit (true sig-eps)
  %
%  epdot_0i = 1.0e8;
%  p_i = 2.0/3.0;
%  q_i = 1.0;
%  if (T < 1040)
%    FisherRawData = [[30   8.6760e+08   3.3102e+00];...
%                     [38   1.5287e+09   4.1211e-01];...
%                     [45   1.6511e+09   1.0482e+00];...
%                     [49   1.7705e+09   1.2546e+00]];
%    if (Rc == 30)
%      sigma_i = 8.6760e+02;
%      g_0i = 3.3102e+00;
%    elseif (Rc == 38)
%      sigma_i = 1.5287e+03;
%      g_0i = 4.1211e-01;
%    elseif (Rc == 45)
%      sigma_i = 1.6511e+03;
%      g_0i = 1.0482e+00;
%    elseif (Rc == 49)
%      sigma_i = 1.7705e+03;
%      g_0i = 1.2546e+00;
%    else
%      g_0i    = -1.7778e-03*Rc^3 + 2.3110e-01*Rc^2 - 9.8832e+00*Rc + 1.3982e+02;
%      sigma_i =  0.2887*Rc^3 - 36.854*Rc^2 + 1586.3*Rc - 21322;
%    end
%  else
%    g_0i = 0.57563;
%    sigma_i = 896.32;
%  end
%  sigma_i = sigma_i*1.0e6;

  %
  % Multiple temperature fit (sig-eps)
  %
  if (T > 1040)
    g_0i = 0.57582;
    sigma_i = 896.14;
  else
    FisherRawData = [[30   8.6760e+08   3.3102e+00];...
                     [38   1.5281e+09   4.1229e-01];...
                     [45   1.6366e+09   1.0460e+00];...
                     [49   1.7521e+09   1.2648e+00]];
    if (Rc == 30)
      sigma_i = 8.6760e+02;
      g_0i = 3.3102e+00;
    elseif (Rc == 38)
      sigma_i = 1.5281e+03;
      g_0i = 4.1229e-01;
    elseif (Rc == 45)
      sigma_i = 1.6366e+03;
      g_0i = 1.0460e+00;
    elseif (Rc == 49)
      sigma_i = 1.7521e+03;
      g_0i = 1.2648e+00;
    else
      g_0i    = -1.7601e-03*Rc^3 + 2.2908e-01*Rc^2 - 9.8074e+00*Rc + 1.3888e+02;
      sigma_i =  0.29934*Rc^3 - 38.296*Rc^2 + 1643.8*Rc - 22062;
    end
  end
  sigma_i = sigma_i*1.0e6;

