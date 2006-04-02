function calcSige

  %
  % Plot the tangent modulus vs sigma_e to find sigma_es
  %
  %plotTangentModulus;

  %
  % Plot modified Arrhenius plots for sigma_es
  %
  plotArrhenius;

%
% Plot modified Arrhenius plots for sigma_es
%
function plotArrhenius

  %
  % Rc = 30
  %
  [datRc30] = sigma_es_Rc30(1);
  [C30Raw] = computePlotArrhenius(datRc30, '4340 Steel Rc 30 (Raw)' );
  [datRc30] = sigma_es_Rc30(2);
  [C30Shf] = computePlotArrhenius(datRc30, '4340 Steel Rc 30 (Shifted)' );
  oog_0es_Rc30_Raw = -C30Raw(1);
  lnsig_0es_Rc30_Raw = C30Raw(2);
  g_0es_Rc30_Raw = 1.0/oog_0es_Rc30_Raw;
  sig_0es_Rc30_Raw = exp(lnsig_0es_Rc30_Raw);
  oog_0es_Rc30_Shf = -C30Shf(1);
  lnsig_0es_Rc30_Shf = C30Shf(2);
  g_0es_Rc30_Shf = 1.0/oog_0es_Rc30_Shf;
  sig_0es_Rc30_Shf = exp(lnsig_0es_Rc30_Shf);

  %
  % Rc = 38
  %
  [datRc38] = sigma_es_Rc38(1);
  [C38Raw] = computePlotArrhenius(datRc38, '4340 Steel Rc 38 (Raw)' );
  [datRc38] = sigma_es_Rc38(2);
  [C38Shf] = computePlotArrhenius(datRc38, '4340 Steel Rc 38 (Shifted)' );
  oog_0es_Rc38_Raw = -C38Raw(1);
  lnsig_0es_Rc38_Raw = C38Raw(2);
  g_0es_Rc38_Raw = 1.0/oog_0es_Rc38_Raw;
  sig_0es_Rc38_Raw = exp(lnsig_0es_Rc38_Raw);
  oog_0es_Rc38_Shf = -C38Shf(1);
  lnsig_0es_Rc38_Shf = C38Shf(2);
  g_0es_Rc38_Shf = 1.0/oog_0es_Rc38_Shf;
  sig_0es_Rc38_Shf = exp(lnsig_0es_Rc38_Shf);

  %
  % Rc = 45
  %
  [datRc45] = sigma_es_Rc45(1);
  [C45Raw] = computePlotArrhenius(datRc45, '4340 Steel Rc 45 (Raw)' );
  [datRc45] = sigma_es_Rc45(2);
  [C45Shf] = computePlotArrhenius(datRc45, '4340 Steel Rc 45 (Shifted)' );
  oog_0es_Rc45_Raw = -C45Raw(1);
  lnsig_0es_Rc45_Raw = C45Raw(2);
  g_0es_Rc45_Raw = 1.0/oog_0es_Rc45_Raw;
  sig_0es_Rc45_Raw = exp(lnsig_0es_Rc45_Raw);
  oog_0es_Rc45_Shf = -C45Shf(1);
  lnsig_0es_Rc45_Shf = C45Shf(2);
  g_0es_Rc45_Shf = 1.0/oog_0es_Rc45_Shf;
  sig_0es_Rc45_Shf = exp(lnsig_0es_Rc45_Shf);

  %
  % Rc = 49
  %
  [datRc49] = sigma_es_Rc49(1);
  [C49Raw] = computePlotArrhenius(datRc49, '4340 Steel Rc 49 (Raw)' );
  [datRc49] = sigma_es_Rc49(2);
  [C49Shf] = computePlotArrhenius(datRc49, '4340 Steel Rc 49 (Shifted)' );
  oog_0es_Rc49_Raw = -C49Raw(1);
  lnsig_0es_Rc49_Raw = C49Raw(2);
  g_0es_Rc49_Raw = 1.0/oog_0es_Rc49_Raw;
  sig_0es_Rc49_Raw = exp(lnsig_0es_Rc49_Raw);
  oog_0es_Rc49_Shf = -C49Shf(1);
  lnsig_0es_Rc49_Shf = C49Shf(2);
  g_0es_Rc49_Shf = 1.0/oog_0es_Rc49_Shf;
  sig_0es_Rc49_Shf = exp(lnsig_0es_Rc49_Shf);

  %
  % Sort with Rc
  %
  Rc = [30 38 45 49];
  g_0es_Raw = [g_0es_Rc30_Raw g_0es_Rc38_Raw g_0es_Rc45_Raw g_0es_Rc49_Raw]
  sig_0es_Raw = [sig_0es_Rc30_Raw sig_0es_Rc38_Raw sig_0es_Rc45_Raw sig_0es_Rc49_Raw]
  g_0es_Shf = [g_0es_Rc30_Shf g_0es_Rc38_Shf g_0es_Rc45_Shf g_0es_Rc49_Shf]
  sig_0es_Shf = [sig_0es_Rc30_Shf sig_0es_Rc38_Shf sig_0es_Rc45_Shf sig_0es_Rc49_Shf]

  %
  % Plot with Rc
  %
  figure;
  subplot(2,2,1);
  p1 = plot(Rc, g_0es_Raw, 'ro'); hold on;
  subplot(2,2,2);
  p2 = plot(Rc, sig_0es_Raw, 'ro'); hold on;
  subplot(2,2,3);
  p3 = plot(Rc, g_0es_Shf, 'ro'); hold on;
  subplot(2,2,4);
  p4 = plot(Rc, sig_0es_Shf, 'ro'); hold on;

  %
  % Fit polygon
  %
  RcMin = 25;
  RcMax = 50;
  nRc = 100;
  dRc = (RcMax-RcMin)/nRc;
  for i=1:nRc+1
    RcFit(i) = RcMin + (i-1)*dRc;
  end
  [pFit1] = polyfit(Rc, g_0es_Raw, 1);
  g_0es_RawFit = polyval(pFit1, RcFit);
  [pFit2] = polyfit(Rc, sig_0es_Raw, 1);
  sig_0es_RawFit = polyval(pFit2, RcFit);
  [pFit3] = polyfit(Rc, g_0es_Shf, 1)
  g_0es_ShfFit = polyval(pFit3, RcFit);
  [pFit4] = polyfit(Rc, sig_0es_Shf, 1)
  sig_0es_ShfFit = polyval(pFit4, RcFit);
  subplot(2,2,1);
  p11 = plot(RcFit, g_0es_RawFit, 'b-');
  subplot(2,2,2);
  p12 = plot(RcFit, sig_0es_RawFit, 'b-');
  subplot(2,2,3);
  p13 = plot(RcFit, g_0es_ShfFit, 'b-');
  subplot(2,2,4);
  p14 = plot(RcFit, sig_0es_ShfFit, 'b-');
  
%
% Compute and plot  Arrhenius plot
%
function [C] = computePlotArrhenius(dat, label)

  fig00 = figure;

  rho0 = 7830.0;
  kappa = 1.3806503e-23;
  b = 2.48e-10;
  edot_0es = 1.0e7;

  edot = dat(:,1);
  T = dat(:,2);
  sig_es = dat(:,3)*1.0e6;
  Tmax = dat(:,4);

  for i=1:length(T)
    
    %
    % Compute ln(sigma_es)
    %
    yy(i) = log(sig_es(i));

    %
    % Compute mu
    %
    rho = rho0;
    P = calcP(rho, rho0, Tmax(i), T(i));
    Tm = calcTm(rho, rho0);
    mu = calcmu(rho, rho0, Tm, P, Tmax(i));

    %
    % Compute kT/mub^3 ln(edot_0es/edot)
    %
    xx(i) = kappa*Tmax(i)/(mu*b^3)*log(edot_0es/edot(i));
    str = sprintf('(%g, %g)',T(i),edot(i));
    text(xx(i),yy(i),str); hold on;
  end
  p1 = plot(xx, yy, 'rs'); hold on;
  set(p1, 'LineWidth', 2, 'MarkerSize', 8);

  [C] = polyfit(xx, yy, 1);
  xmax = 0.2;
  xmin = 0.0;
  nx = 100;
  dx = (xmax-xmin)/nx;
  for i=1:nx+1
    xfit(i) = xmin + (i-1)*dx;
    yfit(i) = C(1)*xfit(i) + C(2);
  end
  p2 = plot(xfit, yfit, 'b-');
  set(p2, 'LineWidth', 3, 'MarkerSize', 8);
  str1 = sprintf('y = %g x + %g', C(1), C(2));
  text(min(xx),min(yy),str1, 'FontName', 'bookman', 'FontSize', 14);

  %set(gca, 'XLim', [0 0.2], 'YLim', [0 8000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('kT/\mu b^3 ln(\epsilon_{0es}/\epsilon) (x)', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('ln(\sigma_{es}) (y) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  title(label, 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;

  
%
% The data extracted from the plots in this program
% The data are stored as (epdot, T, sigma_es, Tmax)
% flag = 1 --- raw sigma_e data
% flag = 2 --- shifted sigma_e data
%
function [dat] = sigma_es_Rc30(flag)
  if (flag == 1)
    dat(1,:)  = [0.0002  298 1200 298];
    dat(2,:)  = [0.009   298  360 298];
    dat(3,:)  = [0.1     298  370 298];
    dat(4,:)  = [1.1     298  425 298];
    dat(5,:)  = [570     298 1400 344];
    dat(6,:)  = [604     500  600 532];
    dat(7,:)  = [650     735  400 758];
  else
    dat(1,:)  = [0.0002  298 1000 298];
    dat(2,:)  = [0.009   298  400 298];
    dat(3,:)  = [0.1     298  420 298];
    dat(4,:)  = [1.1     298  400 298];
    dat(5,:)  = [570     298  900 344];
    dat(6,:)  = [604     500  600 532];
    dat(7,:)  = [650     735  400 758];
  end

function [dat] = sigma_es_Rc38(flag)
  if (flag == 1)
    dat(1,:)  = [0.0002  258  800 258];
    dat(2,:)  = [0.0002  298  700 298];
    dat(3,:)  = [0.0002  373 1000 373];
    dat(4,:)  = [500     298  500 320];
    dat(5,:)  = [500     573  400 591];
    dat(6,:)  = [500     773  250 785];
    dat(7,:)  = [1500    298  600 371];
    dat(8,:)  = [1500    573  400 614];
    dat(9,:)  = [1500    973  270 988];
    dat(10,:) = [1500   1173  200 1185];
    dat(11,:) = [1500   1373  240 1381];
    dat(12,:) = [2500    773  350 815];
    dat(13,:) = [2500    973  325 995];
    dat(14,:) = [2500   1173  125 1193];
    dat(15,:) = [2500   1373  260 1388];
  else
    dat(1,:)  = [0.0002  258  700 258];
    dat(2,:)  = [0.0002  298  600 298];
    dat(3,:)  = [0.0002  373  500 373];
    dat(4,:)  = [500     298  600 320];
    dat(5,:)  = [500     573  400 591];
    dat(6,:)  = [500     773  350 785];
    dat(7,:)  = [1500    298  800 371];
    dat(8,:)  = [1500    573  500 614];
    dat(9,:)  = [1500    973  350 988];
    dat(10,:) = [1500   1173  300 1185];
    dat(11,:) = [1500   1373  170 1381];
    dat(12,:) = [2500    773  550 815];
    dat(13,:) = [2500    973  600 995];
    dat(14,:) = [2500   1173  225 1193];
    dat(15,:) = [2500   1373  275 1388];
  end

function [dat] = sigma_es_Rc45(flag)
  if (flag == 1)
    %dat(1,:) = [0.0001 173 250 173];
    %dat(2,:) = [0.0001 298  50 298];
    %dat(3,:) = [0.0001 373 370 373];
    %dat(4,:) = [1000   173 190 211];
    %dat(5,:) = [1000   298 125 327];
    %dat(6,:) = [1000   373 135 397];

    dat(1,:) = [0.0001 173 250 173];
    dat(2,:) = [0.0001 298  50 298];
    dat(3,:) = [1000   173 190 211];
    dat(4,:) = [1000   373 135 397];
  else
    %dat(1,:) = [0.0001 173 300 173];
    %dat(2,:) = [0.0001 298  90 298];
    %dat(3,:) = [0.0001 373 320 373];
    %dat(4,:) = [1000   173 180 211];
    %dat(5,:) = [1000   298  60 327];
    %dat(6,:) = [1000   373 160 397];

    dat(1,:) = [0.0001 173 300 173];
    dat(2,:) = [0.0001 298  90 298];
    dat(3,:) = [1000   173 180 211];
    dat(4,:) = [1000   373 160 397];
  end

function [dat] = sigma_es_Rc49(flag)
  if (flag == 1)
    %dat(1,:) = [0.0001 173 250 173];
    %dat(2,:) = [0.0001 298 270 298];
    %dat(3,:) = [0.0001 373 375 373];
    %dat(4,:) = [1000   173 180 207];
    %dat(5,:) = [1000   298 275 334];
    %dat(6,:) = [1000   373 300 397];

    dat(1,:) = [0.0001 173 250 173];
    dat(2,:) = [0.0001 298 270 298];
    dat(3,:) = [1000   298 275 334];
    dat(4,:) = [1000   373 300 397];
  else
    %dat(1,:) = [0.0001 173 275 173];
    %dat(2,:) = [0.0001 298 260 298];
    %dat(3,:) = [0.0001 373 375 373];
    %dat(4,:) = [1000   173 160 207];
    %dat(5,:) = [1000   298 300 334];
    %dat(6,:) = [1000   373 280 397];

    dat(1,:) = [0.0001 173 275 173];
    dat(2,:) = [0.0001 298 260 298];
    dat(3,:) = [1000   298 300 334];
    dat(4,:) = [1000   373 280 397];
  end

%
% Plot the tangent modulus vs sigma_e to find sigma_es
%
function plotTangentModulus

  plotTangJCRc30;

  plotTangLarsonRc38;
  plotTangLYRc38500;
  plotTangLYRc381500;
  plotTangLYRc382500;

  plotTangChiRc45_0001;
  plotTangChiRc45_1000;

  plotTangChiRc49_0001;
  plotTangChiRc49_1000;


  %====================================================================

function plotTangJCRc30

  fig00 = figure;
  fig01 = figure;
  E = 213.0e9;
  fitDegree = 3;

  %
  % Load experimental data from Johnson-Cook (Rc = 30)
  %

  %
  % 0.002/s 298K
  %
  load FlowSt0001298KJCTen.dat;
  St298K0002 = FlowSt0001298KJCTen;
  epsEx = St298K0002(:,1);
  seqEx = St298K0002(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  Rc = 30.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);

  %
  % 0.009/s 298K
  %
  load FlowSt0009298KJCShear.dat;
  St298K0009 = FlowSt0009298KJCShear;
  epsEx = St298K0009(:,1)/sqrt(3.0);
  seqEx = St298K0009(:,2)*sqrt(3.0);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0;
  epdot = 0.009;
  T = 298.0;
  Rc = 30.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 0.10/s 298K
  %
  load FlowSt010298KJCShear.dat;
  St298K01 = FlowSt010298KJCShear;
  epsEx = St298K01(:,1)/sqrt(3.0);
  seqEx = St298K01(:,2)*sqrt(3.0);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 0.1;
  epdot = 0.1;
  T = 298.0;
  Rc = 30.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1.1/s 298K
  %
  load FlowSt1_1298KJCShear.dat;
  St298K1 = FlowSt1_1298KJCShear;
  epsEx = St298K1(:,1)/sqrt(3.0);
  seqEx = St298K1(:,2)*sqrt(3.0);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 0.01;
  epdot = 1.1;
  T = 298.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'c-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'c-'); hold on;
  set(psigeiso, 'LineWidth', 3);
 
  %
  % 570/s 298K
  %
  load FlowSt570298KJCTen.dat
  epsEx = FlowSt570298KJCTen(:,1);
  seqEx = FlowSt570298KJCTen(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 570.0;
  T = 298.0;
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'm-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'm-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 604/s 500K
  %
  load FlowSt604500KJCTen.dat
  epsEx = FlowSt604500KJCTen(:,1);
  seqEx = FlowSt604500KJCTen(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 604.0;
  T = 500.0;
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'y-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'y-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 650/s 735K
  %
  load FlowSt650735KJCTen.dat
  epsEx = FlowSt650735KJCTen(:,1);
  seqEx = FlowSt650735KJCTen(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 650.0;
  T = 735.0;
  epmax = max(epsEx);
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'k-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'k-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  figure(fig00);
  set(gca, 'XLim', [0 800], 'YLim', [0 8000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 30', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.9], 'YLim', [0 1200] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 30', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;
  %====================================================================

function plotTangLarsonRc38

  fig00 = figure;
  fig01 = figure;
  E = 213.0e9;

  %
  % Load experimental data from Larson (Rc = 38)
  %
  %
  % 0.0002/s 258 K
  %
  load FlowSt0001258KLarson.dat;
  St258K00002 = FlowSt0001258KLarson;
  epsEx = St258K00002(:,1);
  seqEx = St258K00002(:,2)*6.894657;
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 10.0;
  epdot = 0.0002;
  T = 258.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 0.0002/s 298 K
  %
  load FlowSt0001298KLarson.dat;
  St298K00002 = FlowSt0001298KLarson;
  epsEx = St298K00002(:,1);
  seqEx = St298K00002(:,2)*6.894657;
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 10.0;
  epdot = 0.0002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 0.0002/s 373 K
  %
  load FlowSt0001373KLarson.dat;
  St373K00002 = FlowSt0001373KLarson;
  epsEx = St373K00002(:,1);
  seqEx = St373K00002(:,2)*6.894657;
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 10.0;
  epdot = 0.0002;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  figure(fig00);
  set(gca, 'XLim', [0 800], 'YLim', [0 5000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 0.0002/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.8], 'YLim', [0 1000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 0.0002/s', 'FontName', 'bookman', 'FontSize', 16);
  grid minor;
  axis square;

  %====================================================================

function plotTangLYRc38500

  fig00 = figure;
  fig01 = figure;
  E = 213.0e9;

  %
  % 500/s 298K
  %
  load FlowSt500298KLY.dat
  epsEx = FlowSt500298KLY(:,1)*1.0e-2;
  seqEx = FlowSt500298KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 500/s 573K
  %
  load FlowSt500573KLY.dat
  epsEx = FlowSt500573KLY(:,1)*1.0e-2;
  seqEx = FlowSt500573KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 500/s 773K
  %
  load FlowSt500773KLY.dat
  epsEx = FlowSt500773KLY(:,1)*1.0e-2;
  seqEx = FlowSt500773KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  figure(fig00);
  set(gca, 'XLim', [0 500], 'YLim', [0 16000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 500/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.1], 'YLim', [0 400] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 500/s', 'FontName', 'bookman', 'FontSize', 16);
  grid minor
  axis square;
         
  %====================================================================

function plotTangLYRc381500

  fig00 = figure;
  fig01 = figure;
  E = 213.0e9;
  %
  % 1500/s 298K
  %
  load FlowSt1500298KLY.dat
  epsEx = FlowSt1500298KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500298KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 573K
  %
  load FlowSt1500573KLY.dat
  epsEx = FlowSt1500573KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500573KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 973K
  %
  load FlowSt1500973KLY.dat
  epsEx = FlowSt1500973KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500973KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 1173K
  %
  load FlowSt15001173KLY.dat
  epsEx = FlowSt15001173KLY(:,1)*1.0e-2;
  seqEx = FlowSt15001173KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'm-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'm-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 1373K
  %
  load FlowSt15001373KLY.dat
  epsEx = FlowSt15001373KLY(:,1)*1.0e-2;
  seqEx = FlowSt15001373KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'k-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'k-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  figure(fig00);
  set(gca, 'XLim', [0 600], 'YLim', [0 8000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 1500/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square

  figure(fig01);
  set(gca, 'XLim', [0 0.2], 'YLim', [0 600] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 1500/s', 'FontName', 'bookman', 'FontSize', 16);
  grid minor;
  axis square;
         
  %====================================================================
         
function plotTangLYRc382500

  fig00 = figure;
  fig01 = figure;
  E = 213.0e9;
  %
  % 2500/s 773K
  %
  load FlowSt2500773KLY.dat
  epsEx = FlowSt2500773KLY(:,1)*1.0e-2;
  seqEx = FlowSt2500773KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 2500/s 973K
  %
  load FlowSt2500973KLY.dat
  epsEx = FlowSt2500973KLY(:,1)*1.0e-2;
  seqEx = FlowSt2500973KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 2500/s 1173K
  %
  load FlowSt25001173KLY.dat
  epsEx = FlowSt25001173KLY(:,1)*1.0e-2;
  seqEx = FlowSt25001173KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 2500/s 1373K
  %
  load FlowSt25001373KLY.dat
  epsEx = FlowSt25001373KLY(:,1)*1.0e-2;
  seqEx = FlowSt25001373KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'm-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'm-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  figure(fig00);
  set(gca, 'XLim', [0 400], 'YLim', [0 4000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 2500/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.3], 'YLim', [0 400] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 2500/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  grid minor;
  axis square;

  %====================================================================
         
function plotTangChiRc45_0001

  E = 213.0e9;
  fig00 = figure;
  fig01 = figure;
  fig02 = figure;

  %
  % Plot experimental data for 4340 steel Rc 45 (Chi et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load FlowSt0001173KChi.dat
  epsEx = FlowSt0001173KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001173KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'r-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'r-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'r--', 'LineWidth', 2); hold on;

  load FlowSt0001298KChi.dat
  epsEx = FlowSt0001298KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001298KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'g-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'g-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'g--', 'LineWidth', 2); hold on;
  
  load FlowSt0001373KChi.dat
  epsEx = FlowSt0001373KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001373KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'b-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'b-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'b--', 'LineWidth', 2); hold on;
  
  figure(fig00);
  set(gca, 'XLim', [0 400], 'YLim', [0 15000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.25], 'YLim', [0 400] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  grid minor;
  axis square;

  %====================================================================

function plotTangChiRc45_1000

  fig00 = figure;
  fig01 = figure;
  fig02 = figure;
  E = 213.0e9;

  %
  % Plot experimental data for 4340 steel Rc 45 (Chi et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load FlowSt1000173KChi.dat
  epsEx = FlowSt1000173KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000173KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'r-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'r-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'r--', 'LineWidth', 2); hold on;
  
  load FlowSt1000298KChi.dat
  epsEx = FlowSt1000298KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000298KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'g-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'g-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'g--', 'LineWidth', 2); hold on;
  
  load FlowSt1000373KChi.dat
  epsEx = FlowSt1000373KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000373KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'b-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'b-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'b--', 'LineWidth', 2); hold on;
  
  figure(fig00);
  set(gca, 'XLim', [0 200], 'YLim', [0 20000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.07], 'YLim', [0 250] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;

  %====================================================================

function plotTangChiRc49_0001

  fig00 = figure;
  fig01 = figure;
  fig02 = figure;

  E = 213.0e9;

  %
  % Plot experimental data for 4340 steel Rc 49 (Chi2 et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load FlowSt0001173KChi2.dat
  epsEx = FlowSt0001173KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001173KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'r-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'r-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'r--', 'LineWidth', 2); hold on;

  load FlowSt0001298KChi2.dat
  epsEx = FlowSt0001298KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001298KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'g-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'g-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'g--', 'LineWidth', 2); hold on;
  
  load FlowSt0001373KChi2.dat
  epsEx = FlowSt0001373KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001373KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'b-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'b-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'b--', 'LineWidth', 2); hold on;
  
  figure(fig00);
  set(gca, 'XLim', [0 500], 'YLim', [0 25000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.25], 'YLim', [0 450] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;

  %====================================================================

function plotTangChiRc49_1000

  fig00 = figure;
  fig01 = figure;
  fig02 = figure;

  E = 213.0e9;

  %
  % Plot experimental data for 4340 steel Rc 49 (Chi2 et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load FlowSt1000173KChi2.dat
  epsEx = FlowSt1000173KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000173KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'r-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'r-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'r-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'r-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'r--', 'LineWidth', 2); hold on;
  
  load FlowSt1000298KChi2.dat
  epsEx = FlowSt1000298KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000298KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'g-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'g-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'g-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'g-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'g--', 'LineWidth', 2); hold on;
  
  load FlowSt1000373KChi2.dat
  epsEx = FlowSt1000373KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000373KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E - 0.003;
  epEx = epEx - epEx(2);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'b-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'b-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'b-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'b-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'b--', 'LineWidth', 2); hold on;
  
  figure(fig00);
  set(gca, 'XLim', [0 400], 'YLim', [0 30000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  figure(fig01);
  set(gca, 'XLim', [0 0.07], 'YLim', [0 350] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
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
% Compute the tangent modulus and sigma_e (isothermal)
%
function [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
          computeTangentModulusIso(delT, ep, sig, epdot, T0, Rc)

  [n, m] = size(ep);
  count = 1;
  for i=1:n
    if ~(ep(i) < 0.0)
      sig_m = sig(i);
      [sig_e(count), sig_i(count),mu0(count),mu(count)] = ...
         computeSige(sig_m, epdot, T0, T0, Rc);
      eplas(count) = ep(i);
      count = count+1;
    end
  end
  T0
  sig_e = sig_e - sig_e(1);
  n = length(eplas);
  count = 1;
  for i=1:n-1
    dep = eplas(i+1) - eplas(i);
    dsig = sig_e(i+1) - sig_e(i);
    theta(count) = dsig/dep;
    sigma_e(count) = 0.5*(sig_e(i+1) + sig_e(i));
    count = count + 1;
  end

%
% Compute the tangent modulus and sigma_e (adiabatic)
%
function [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
          computeTangentModulusAdi(delT, ep, sig, epdot, T0, Rc)

  [n, m] = size(ep);
  count = 1;
  for i=1:n
    if ~(ep(i) < 0.0)
      sig_m = sig(i);
      ep_m =  ep(i);
      T = computeTemp(delT, sig_m, ep_m, epdot, T0);
      [sig_e(count), sig_i(count),mu0(count),mu(count)] = ...
         computeSige(sig_m, epdot, T, T0, Rc);
      eplas(count) = ep(i);
      count = count+1;
    end
  end
  T
  sig_e = sig_e - sig_e(1);
  n = length(eplas);
  count = 1;
  for i=1:n-1
    dep = eplas(i+1) - eplas(i);
    dsig = sig_e(i+1) - sig_e(i);
    if (dep ~= 0.0)
      theta(count) = dsig/dep;
      sigma_e(count) = 0.5*(sig_e(i+1) + sig_e(i));
      count = count + 1;
    end
  end

%
% Compute the adiabatic temperature
%
function [T] = computeTemp(delT, sig, ep, epdot, T0)

  rho0 = 7830.0;
  tmax = ep/epdot;
  m = tmax/delT;
  rho = rho0;
  T = T0;
  ep = 0.0;
  for i=1:m
    Cp = calcCp(T);
    fac = 0.9/(rho*Cp);
    T = T + sig*epdot*fac*delT; 
  end

%
% Compute sigma_e from stress-strain plot
%
function [sigma_e, sigma_i, mu_0, mu] = computeSige(sig_y, epdot, T, T0, Rc)

  %
  % Compute mu_0
  %
  rho0 = 7830.0;
  P = calcP(rho0, rho0, 0, 0);
  Tm = calcTm(rho0, rho0);
  mu_0 = calcmu(rho0, rho0, Tm, P, 0);

  %
  % Compute mu
  %
  rho = rho0;
  P = calcP(rho, rho0, T, T0);
  Tm = calcTm(rho, rho0);
  mu = calcmu(rho, rho0, Tm, P, T);

  %
  % Compute S_i
  %
  kappa = 1.3806503e-23;
  b = 2.48e-10;
  sig_a = 50.0e6;
  edot_0i = 1.0e8;
  p_i = 2.0/3.0;
  q_i = 1.0;
  g_0i = -1.5425e-3*Rc^3 + 2.0396e-1*Rc^2 - 8.8536*Rc + 1.27e2;
  sigma_i = 0.18162*Rc^3 - 24.029*Rc^2 + 1077.1*Rc - 14721;
  sigma_i = sigma_i*1.0e6;
  S_i = (1.0 - (kappa*T/(mu*b^3*g_0i)*log(edot_0i/epdot))^(1/q_i))^(1/p_i);

  %
  % Compute S_e
  %
  kappa = 1.3806503e-23;
  b = 2.48e-10;
  edot_0e = 1.0e7;
  p_e = 2.0/3.0;
  q_e = 1.0;
  g_0e = 1.6;
  %p_e = 0.5;
  %q_e = 1.5;
  %g_0e = 1.0;
  S_e = (1.0 - (kappa*T/(mu*b^3*g_0e)*log(edot_0e/epdot))^(1/q_e))^(1/p_e);

  %
  % Compute sig_e
  %
  sigma_e = (1.0/S_e)*(mu_0/mu*(sig_y - sig_a) - S_i*sigma_i);
  
  %[sigma_e mu_0 mu sig_y sig_a S_i sigma_i]
  sigma_i = S_i*sigma_i;
