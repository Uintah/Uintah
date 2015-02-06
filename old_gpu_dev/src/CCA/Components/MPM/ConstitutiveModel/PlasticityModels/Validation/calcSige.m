function calcSige

  %
  % Plot the tangent modulus vs sigma_e to find sigma_es
  %
  plotTangentModulus;

  %
  % Plot modified Arrhenius plots for sigma_es
  %
  %plotArrhenius;

%
% Plot modified Arrhenius plots for sigma_es
%
function plotArrhenius

  format short e;

  %
  % Rc = 30
  %
  %[datRc30] = sigma_es_Rc30(1);
  %[C30Raw] = computePlotArrhenius(datRc30, '4340 Steel Rc 30 (Raw)', ...
  %                                'StDataSigeFisherRc30.dat' );
  %oog_0es_Rc30_Raw = -C30Raw(1);
  %lnsig_0es_Rc30_Raw = C30Raw(2);
  %g_0es_Rc30_Raw = 1.0/oog_0es_Rc30_Raw;
  %sig_0es_Rc30_Raw = exp(lnsig_0es_Rc30_Raw);

  [datRc30] = sigma_es_Rc30(4);
  [C30Shf] = computePlotArrhenius(datRc30, '4340 Steel Rc 30', ...
                                  'StDataSigeFisherRc30.dat');
  oog_0es_Rc30_Shf = -C30Shf(1);
  lnsig_0es_Rc30_Shf = C30Shf(2);
  g_0es_Rc30_Shf = 1.0/oog_0es_Rc30_Shf;
  sig_0es_Rc30_Shf = exp(lnsig_0es_Rc30_Shf);

  %
  % Rc = 38
  %
  %[datRc38,datLo38,datHi38] = sigma_es_Rc38(1);
  %[C38Raw] = computePlotArrhenius(datRc38, '4340 Steel Rc 38 (Raw)' , ...
  %                                'StDataSigeFisherRc38.dat');
  %oog_0es_Rc38_Raw = -C38Raw(1);
  %lnsig_0es_Rc38_Raw = C38Raw(2);
  %g_0es_Rc38_Raw = 1.0/oog_0es_Rc38_Raw;
  %sig_0es_Rc38_Raw = exp(lnsig_0es_Rc38_Raw);

  [datRc38,datLo38,datHi38] = sigma_es_Rc38(4);
  [C38Shf] = computePlotArrhenius(datRc38, '4340 Steel Rc 38' , ...
                                  'StDataSigeFisherRc38.dat');
  oog_0es_Rc38_Shf = -C38Shf(1);
  lnsig_0es_Rc38_Shf = C38Shf(2);
  g_0es_Rc38_Shf = 1.0/oog_0es_Rc38_Shf;
  sig_0es_Rc38_Shf = exp(lnsig_0es_Rc38_Shf);

  [CLoShf] = computePlotArrhenius(datLo38, '4340 Steel Rc 38 Low T', ...
                                  'StDataSigeFisherRc38Lo.dat' );
  oog_0es_Lo38_Shf = -CLoShf(1);
  lnsig_0es_Lo38_Shf = CLoShf(2);
  g_0es_Lo38_Shf = 1.0/oog_0es_Lo38_Shf;
  sig_0es_Lo38_Shf = exp(lnsig_0es_Lo38_Shf);

  [CHiShf] = computePlotArrhenius(datHi38, '4340 Steel Rc 38 High T' , ...
                                  'StDataSigeFisherRc38Hi.dat');
  oog_0es_Hi38_Shf = -CHiShf(1);
  lnsig_0es_Hi38_Shf = CHiShf(2);
  g_0es_Hi38_Shf = 1.0/oog_0es_Hi38_Shf
  sig_0es_Hi38_Shf = exp(lnsig_0es_Hi38_Shf)

  %
  % Rc = 45
  %
  %[datRc45] = sigma_es_Rc45(1);
  %[C45Raw] = computePlotArrhenius(datRc45, '4340 Steel Rc 45 (Raw)' , ...
  %                                'StDataSigeFisherRc45.dat');
  %oog_0es_Rc45_Raw = -C45Raw(1);
  %lnsig_0es_Rc45_Raw = C45Raw(2);
  %g_0es_Rc45_Raw = 1.0/oog_0es_Rc45_Raw;
  %sig_0es_Rc45_Raw = exp(lnsig_0es_Rc45_Raw);

  [datRc45] = sigma_es_Rc45(4);
  [C45Shf] = computePlotArrhenius(datRc45, '4340 Steel Rc 45' , ...
                                  'StDataSigeFisherRc45.dat');
  oog_0es_Rc45_Shf = -C45Shf(1);
  lnsig_0es_Rc45_Shf = C45Shf(2);
  g_0es_Rc45_Shf = 1.0/oog_0es_Rc45_Shf;
  sig_0es_Rc45_Shf = exp(lnsig_0es_Rc45_Shf);

  %
  % Rc = 49
  %
  %[datRc49] = sigma_es_Rc49(1);
  %[C49Raw] = computePlotArrhenius(datRc49, '4340 Steel Rc 49 (Raw)' , ...
  %                                'StDataSigeFisherRc49.dat');
  %oog_0es_Rc49_Raw = -C49Raw(1);
  %lnsig_0es_Rc49_Raw = C49Raw(2);
  %g_0es_Rc49_Raw = 1.0/oog_0es_Rc49_Raw;
  %sig_0es_Rc49_Raw = exp(lnsig_0es_Rc49_Raw);

  [datRc49] = sigma_es_Rc49(4);
  [C49Shf] = computePlotArrhenius(datRc49, '4340 Steel Rc 49' , ...
                                  'StDataSigeFisherRc49.dat');
  oog_0es_Rc49_Shf = -C49Shf(1);
  lnsig_0es_Rc49_Shf = C49Shf(2);
  g_0es_Rc49_Shf = 1.0/oog_0es_Rc49_Shf;
  sig_0es_Rc49_Shf = exp(lnsig_0es_Rc49_Shf);

  %
  % Sort with Rc
  %
  %Rc = [30 38 45 49];
  %g_0es_Raw = [g_0es_Rc30_Raw g_0es_Rc38_Raw g_0es_Rc45_Raw g_0es_Rc49_Raw]
  %sig_0es_Raw = [sig_0es_Rc30_Raw sig_0es_Rc38_Raw sig_0es_Rc45_Raw sig_0es_Rc49_Raw]

  Rc = [30 38 45 49];
  g_0es_Shf   = [g_0es_Rc30_Shf   g_0es_Lo38_Shf   g_0es_Rc45_Shf   g_0es_Rc49_Shf]
  sig_0es_Shf = [sig_0es_Rc30_Shf sig_0es_Lo38_Shf sig_0es_Rc45_Shf sig_0es_Rc49_Shf]

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
  %[pFit1] = polyfit(Rc, g_0es_Raw, 1);
  %g_0es_RawFit = polyval(pFit1, RcFit);
  %[pFit2] = polyfit(Rc, sig_0es_Raw, 1);
  %sig_0es_RawFit = polyval(pFit2, RcFit);
  [pFit3] = polyfit(Rc, g_0es_Shf, 3)
  g_0es_ShfFit = polyval(pFit3, RcFit);
  [pFit4] = polyfit(Rc, sig_0es_Shf, 3)
  sig_0es_ShfFit = polyval(pFit4, RcFit);

  %
  % Plot with Rc (Rawdata)
  %
  %subplot(2,2,1);
  %p1 = plot(Rc, g_0es_Raw, 'ro', 'LineWidth', 2, 'MarkerSize', 8); hold on;
  %subplot(2,2,2);
  %p2 = plot(Rc, sig_0es_Raw, 'ro', 'LineWidth', 2, 'MarkerSize', 8); hold on;
  %subplot(2,2,1);
  %p11 = plot(RcFit, g_0es_RawFit, 'b-', 'LineWidth', 3, 'MarkerSize', 8);
  %subplot(2,2,2);
  %p12 = plot(RcFit, sig_0es_RawFit, 'b-', 'LineWidth', 3, 'MarkerSize', 8);

  %
  % Plot with Rc (Shifted data)
  %
  fig1 = figure;
  p3 = plot(Rc, g_0es_Shf, 'ro', 'LineWidth', 2, 'MarkerSize', 8); hold on;
  p13 = plot(RcFit, g_0es_ShfFit, 'b-', 'LineWidth', 3, 'MarkerSize', 8);
  set(gca, 'XLim', [25 50], 'YLim', [0 0.4] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Hardness (R_c)', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('g_{0es} ', ...
         'FontName', 'bookman', 'FontSize', 16);
  axis square;
  str_g0es = sprintf('g_{0es} = %0.5g R_c - %0.5g', pFit3(1), -pFit3(2));
  gtext(str_g0es, 'FontName', 'bookman', 'FontSize', 14);

  fig2 = figure;
  p4 = plot(Rc, sig_0es_Shf*1.0e-6, 'ro', 'LineWidth', 2, 'MarkerSize', 8); hold on;
  p14 = plot(RcFit, sig_0es_ShfFit*1.0e-6, 'b-', 'LineWidth', 3, 'MarkerSize', 8);
  set(gca, 'XLim', [25 50], 'YLim', [0 1600] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('Hardness (R_c)', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_{0es} (MPa) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  axis square;
  pFit4 = pFit4*1.0e-6;
  str_sig0es = sprintf('\\sigma_{0es} = %0.5g R_c + %0.5g (MPa)', pFit4(1), pFit4(2))
  gtext(str_sig0es, 'FontName', 'bookman', 'FontSize', 14);
  
%
% Compute and plot  Arrhenius plot
%
function [C] = computePlotArrhenius(dat, label, fileName)

  fig00 = figure;

  rho0 = 7830.0;
  kappa = 1.3806503e-23;
  b = 2.48e-10;
  edot_0es = 1.0e7;

  edot = dat(:,1);
  T = dat(:,2);
  sig_es = dat(:,3)*1.0e6;
  Tmax = dat(:,4);

  fid = fopen(fileName', 'w');
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
    %str = sprintf('(%g, %g)',T(i),edot(i));
    %text(xx(i),yy(i),str); hold on;
    fprintf(fid, '%0.5g %0.5g %0.5g %0.5g %0.4g %0.5g %0.5g\n', ...
            xx(i), yy(i), T(i), Tmax(i), edot(i), sig_es(i)*1.0e-6, mu*1.0e-9);
  end
  fclose(fid);

  p1 = plot(xx, yy, 'rs'); hold on;
  set(p1, 'LineWidth', 2, 'MarkerSize', 8);

  [C, S] = polyfit(xx, yy, 1);
  %xmax = max(xx)+0.05;
  xmin = 0.0;
  xmax = 0.3;
  nx = 3;
  dx = (xmax-xmin)/nx;
  for i=1:nx+1
    xfit(i) = xmin + (i-1)*dx;
  end
  [yfit, delta]  = polyval(C, xfit, S);
  p2 = plot(xfit, yfit, 'b-');
  %p2 = errorbar(xfit, yfit, delta, 'b-');
  set(p2, 'LineWidth', 3, 'MarkerSize', 8);
  str1 = sprintf('y = %g x + %g (ln(Pa))', C(1), C(2));
  text(min(xx),min(yy),str1, 'FontName', 'bookman', 'FontSize', 14);

  set(gca, 'XLim', [0 0.3], 'YLim', [17 22] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('x := kT/\mu b^3 ln(\epsilon_{0es}/\epsilon)', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('y := ln(\sigma_{es}) (ln(Pa))', ...
         'FontName', 'bookman', 'FontSize', 16);
  title(label, 'FontName', 'bookman', 'FontSize', 16);
  %grid on;
  axis square;

  
%
% The data extracted from the plots in this program
% The data are stored as (epdot, T, sigma_es, Tmax)
% flag = 1 --- raw Best Fit sigma_e data
% flag = 2 --- shifted Best Fit sigma_e data
% flag = 3 --- shifted HY-100 based sigma_e data
% flag = 4 --- shifted multiple T based sigma_e data
%
function [dat] = sigma_es_Rc30(flag)
  if (flag == 1)
    dat(1,:)  = [0.002  298 1200 298];
    dat(2,:)  = [0.009   298  360 298];
    dat(3,:)  = [0.1     298  370 298];
    dat(4,:)  = [1.1     298  425 298];
    dat(5,:)  = [570     298 1400 344];
    dat(6,:)  = [604     500  600 532];
    dat(7,:)  = [650     735  400 758];
  elseif (flag == 2)
    dat(1,:)  = [0.002  298 1000 298];
    dat(2,:)  = [0.009   298  400 298];
    dat(3,:)  = [0.1     298  420 298];
    dat(4,:)  = [1.1     298  400 298];
    dat(5,:)  = [570     298  900 344];
    dat(6,:)  = [604     500  600 532];
    dat(7,:)  = [650     735  400 758];
  elseif (flag == 3)
    dat(1,:)  = [0.002  298 1500 298];
    dat(2,:)  = [0.009   298  400 298];
    dat(3,:)  = [0.1     298  500 298];
    dat(4,:)  = [1.1     298  400 298];
    dat(5,:)  = [570     298 1400 344];
    dat(6,:)  = [604     500  750 532];
    dat(7,:)  = [650     735  400 758];
  else
    dat(1,:)  = [0.002   298 1500 298];
    dat(2,:)  = [0.009   298  360 298];
    dat(3,:)  = [0.1     298  450 298];
    dat(4,:)  = [1.1     298  470 298];
    dat(5,:)  = [570     298 1000 344];
    dat(6,:)  = [604     500  650 532];
    dat(7,:)  = [650     735  400 758];
  end

function [dat, datLo, datHi] = sigma_es_Rc38(flag)
  if (flag == 1)
    datLo(1,:)  = [0.0002  258  800 258];
    datLo(2,:)  = [0.0002  298  700 298];
    datLo(3,:)  = [0.0002  373 1000 373];
    datLo(4,:)  = [500     298  500 320];
    datLo(5,:)  = [1500    298  600 371];
    datLo(6,:)  = [500     573  400 591];
    datLo(7,:)  = [500     773  250 785];
    datLo(8,:)  = [1500    573  400 614];
    datLo(9,:)  = [1500    973  270 988];
    datLo(10,:) = [2500    773  350 815];
    datLo(11,:) = [2500    973  325 995];
    datHi(1,:) = [1500   1173  200 1185];
    datHi(2,:) = [1500   1373  240 1381];
    datHi(3,:) = [2500   1173  125 1193];
    datHi(4,:) = [2500   1373  260 1388];
  elseif (flag == 2)
    datLo(1,:)  = [0.0002  258  700 258];
    datLo(2,:)  = [0.0002  298  600 298];
    datLo(3,:)  = [0.0002  373  500 373];
    datLo(4,:)  = [500     298  600 320];
    datLo(5,:)  = [1500    298  800 371];
    datLo(6,:)  = [500     573  400 591];
    datLo(7,:)  = [500     773  350 785];
    datLo(8,:)  = [1500    573  500 614];
    datLo(9,:)  = [1500    973  350 988];
    datLo(10,:) = [2500    773  550 815];
    datLo(11,:) = [2500    973  600 995];
    datHi(1,:) = [1500   1173  300 1185];
    datHi(2,:) = [1500   1373  170 1381];
    datHi(3,:) = [2500   1173  225 1193];
    datHi(4,:) = [2500   1373  275 1388];
  elseif (flag == 3)
    datLo(1,:)  = [0.0002  258  900 258];
    datLo(2,:)  = [0.0002  298  800 298];
    datLo(3,:)  = [0.0002  373  800 373];
    datLo(4,:)  = [500     298  650 320];
    datLo(5,:)  = [1500    298  950 371];
    datLo(6,:)  = [500     573  550 591];
    datLo(7,:)  = [500     773  300 785];
    datLo(8,:)  = [1500    573  800 614];
    datLo(9,:)  = [1500    973  500 988];
    datLo(10,:) = [2500    773  480 815];
    datLo(11,:) = [2500    973  450 995];
    datHi(1,:) = [1500   1173  400 1185];
    datHi(2,:) = [1500   1373  160 1381];
    datHi(3,:) = [2500   1173  160 1193];
    datHi(4,:) = [2500   1373  200 1388];
  else
    datLo(1,:)  = [0.0002  258  1100 258];
    datLo(2,:)  = [0.0002  298  850 298];
    datLo(3,:)  = [0.0002  373  2000 373];
    datLo(4,:)  = [500     298  650 320];
    datLo(5,:)  = [500     573  580 591];
    datLo(6,:)  = [500     773  270 785];
    datLo(7,:)  = [1500    298  1150 371];
    datLo(8,:)  = [1500    573  1050 614];
    datLo(9,:)  = [1500    973  550 988];
    datLo(10,:) = [2500    773  650 815];
    datLo(11,:) = [2500    973  450 995];
    datHi(1,:) =  [1500   1173  250 1185];
    datHi(2,:) =  [1500   1373  150 1381];
    datHi(3,:) =  [2500   1173  200 1193];
    datHi(4,:) =  [2500   1373  270 1388];
  end
  dat = cat(1,datLo,datHi);

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
  elseif (flag == 2)
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
  elseif (flag == 3)
    %dat(1,:) = [0.0001 173 305 173];
    %dat(2,:) = [0.0001 298  90 298];
    %dat(3,:) = [0.0001 373 320 373];
    %dat(4,:) = [1000   173 182 211];
    %dat(5,:) = [1000   298  55 327];
    %dat(6,:) = [1000   373 150 397];

    dat(1,:) = [0.0001 173 305 173];
    dat(2,:) = [0.0001 298  90 298];
    dat(3,:) = [1000   173 182 211];
    dat(4,:) = [1000   373 150 397];
  else
    dat(1,:) = [0.0001 173 260 173];
    dat(2,:) = [0.0001 298  25 298];
    dat(3,:) = [0.0001 373 375 373];
    dat(4,:) = [1000   173 200 211];
    dat(5,:) = [1000   373 130 397];
    dat(6,:) = [1000   298 140 327];
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
  elseif (flag == 2)
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
  elseif (flag == 3)
    %dat(1,:) = [0.0001 173 280 173];
    %dat(2,:) = [0.0001 298 260 298];
    %dat(3,:) = [0.0001 373 380 373];
    %dat(4,:) = [1000   173 157 207];
    %dat(5,:) = [1000   298 298 334];
    %dat(6,:) = [1000   373 310 397];

    dat(1,:) = [0.0001 173 280 173];
    dat(2,:) = [0.0001 298 260 298];
    dat(3,:) = [1000   298 298 334];
    dat(4,:) = [1000   373 310 397];
  else
    dat(1,:) = [0.0001 173 250 173];
    dat(2,:) = [0.0001 298 275 298];
    dat(3,:) = [0.0001 373 370 373];
    dat(4,:) = [1000   173 200 207];
    dat(5,:) = [1000   298 300 334];
    dat(6,:) = [1000   373 350 397];
  end

%
% Plot the tangent modulus vs sigma_e to find sigma_es
%
function plotTangentModulus

  plotTangJCRc30Tension;
  plotTangJCRc30Shear;

  plotTangLarsonRc38;
  plotTangLYRc38500;
  plotTangLYRc381500;
  plotTangLYRc382500;

  plotTangChiRc45_0001;
  plotTangChiRc45_1000;

  plotTangChiRc49_0001;
  plotTangChiRc49_1000;


%====================================================================
%
% Load experimental data from Johnson-Cook (Rc = 30)
%
function plotTangJCRc30Tension

  fig00 = figure;
  fig01 = figure;

  %
  % 0.002/s 298K
  %
  load SigEpsEp0002298KRc30Ten.dat;
  epEx = SigEpsEp0002298KRc30Ten(:,2);
  seqEx = SigEpsEp0002298KRc30Ten(:,3);

  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  Rc = 30.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);

  %
  % 570/s 298K
  %
  load SigEpsEp570298KRc30Ten.dat
  epEx = SigEpsEp570298KRc30Ten(:,2);
  seqEx = SigEpsEp570298KRc30Ten(:,3);

  delT = 1.0e-6;
  epdot = 570.0;
  T = 298.0;
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'mo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'mo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 604/s 500K
  %
  load SigEpsEp604500KRc30Ten.dat
  epEx = SigEpsEp604500KRc30Ten(:,2);
  seqEx = SigEpsEp604500KRc30Ten(:,3);

  delT = 1.0e-6;
  epdot = 604.0;
  T = 500.0;
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'yo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'yo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 650/s 735K
  %
  load SigEpsEp650735KRc30Ten.dat
  epEx = SigEpsEp650735KRc30Ten(:,2);
  seqEx = SigEpsEp650735KRc30Ten(:,3);

  delT = 1.0e-6;
  epdot = 650.0;
  T = 735.0;
  epmax = max(epEx);
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ko-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ko-'); hold on;
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
%
% Load experimental data from Johnson-Cook (Rc = 30)
%
function plotTangJCRc30Shear

  fig00 = figure;
  fig01 = figure;

  %
  % 0.009/s 298K
  %
  load SigEpsEp0009298KRc30Shear.dat;
  epEx = SigEpsEp0009298KRc30Shear(:,2);
  seqEx = SigEpsEp0009298KRc30Shear(:,3);

  delT = 1.0;
  epdot = 0.009;
  T = 298.0;
  Rc = 30.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 0.10/s 298K
  %
  load SigEpsEp010298KRc30Shear.dat;
  epEx = SigEpsEp010298KRc30Shear(:,2);
  seqEx = SigEpsEp010298KRc30Shear(:,3);

  delT = 0.1;
  epdot = 0.1;
  T = 298.0;
  Rc = 30.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1.1/s 298K
  %
  load SigEpsEp1_1298KRc30Shear.dat;
  epEx = SigEpsEp1_1298KRc30Shear(:,2);
  seqEx = SigEpsEp1_1298KRc30Shear(:,3);

  delT = 0.01;
  epdot = 1.1;
  T = 298.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'co-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'co-'); hold on;
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
%
% Load experimental data from Larson (Rc = 38)
%
function plotTangLarsonRc38

  fig00 = figure;
  fig01 = figure;

  %
  % 0.0002/s 258 K
  %
  load SigEpsEp0002258KRc38.dat;
  epEx = SigEpsEp0002258KRc38(:,2);
  seqEx = SigEpsEp0002258KRc38(:,3);

  delT = 10.0;
  epdot = 0.0002;
  T = 258.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 0.0002/s 298 K
  %
  load SigEpsEp0002298KRc38.dat;
  epEx = SigEpsEp0002298KRc38(:,2);
  seqEx = SigEpsEp0002298KRc38(:,3);

  delT = 10.0;
  epdot = 0.0002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 0.0002/s 373 K
  %
  load SigEpsEp0002373KRc38.dat;
  epEx = SigEpsEp0002373KRc38(:,2);
  seqEx = SigEpsEp0002373KRc38(:,3);

  delT = 10.0;
  epdot = 0.0002;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epEx);
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
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

  %
  % 500/s 298K
  %
  load SigEpsEp500298KRc38.dat
  epEx = SigEpsEp500298KRc38(:,2);
  seqEx = SigEpsEp500298KRc38(:,3);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 500/s 573K
  %
  load SigEpsEp500573KRc38.dat
  epEx = SigEpsEp500573KRc38(:,2);
  seqEx = SigEpsEp500573KRc38(:,3);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 500/s 773K
  %
  load SigEpsEp500773KRc38.dat
  epEx = SigEpsEp500773KRc38(:,2);
  seqEx = SigEpsEp500773KRc38(:,3);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
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

  %
  % 1500/s 298K
  %
  load SigEpsEp1500298KRc38.dat
  epEx = SigEpsEp1500298KRc38(:,2);
  seqEx = SigEpsEp1500298KRc38(:,3);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 573K
  %
  load SigEpsEp1500573KRc38.dat
  epEx = SigEpsEp1500573KRc38(:,2);
  seqEx = SigEpsEp1500573KRc38(:,3);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 973K
  %
  load SigEpsEp1500973KRc38.dat
  epEx = SigEpsEp1500973KRc38(:,2);
  seqEx = SigEpsEp1500973KRc38(:,3);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 1173K
  %
  load SigEpsEp15001173KRc38.dat
  epEx = SigEpsEp15001173KRc38(:,2);
  seqEx = SigEpsEp15001173KRc38(:,3);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'mo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'mo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 1500/s 1373K
  %
  load SigEpsEp15001373KRc38.dat
  epEx = SigEpsEp15001373KRc38(:,2);
  seqEx = SigEpsEp15001373KRc38(:,3);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ko-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ko-'); hold on;
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

  %
  % 2500/s 773K
  %
  load SigEpsEp2500773KRc38.dat
  epEx = SigEpsEp2500773KRc38(:,2);
  seqEx = SigEpsEp2500773KRc38(:,3);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 2500/s 973K
  %
  load SigEpsEp2500973KRc38.dat
  epEx = SigEpsEp2500973KRc38(:,2);
  seqEx = SigEpsEp2500973KRc38(:,3);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 2500/s 1173K
  %
  load SigEpsEp25001173KRc38.dat
  epEx = SigEpsEp25001173KRc38(:,2);
  seqEx = SigEpsEp25001173KRc38(:,3);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  
  %
  % 2500/s 1373K
  %
  load SigEpsEp25001373KRc38.dat
  epEx = SigEpsEp25001373KRc38(:,2);
  seqEx = SigEpsEp25001373KRc38(:,3);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epEx);
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'mo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'mo-'); hold on;
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

  fig00 = figure;
  fig01 = figure;
  fig02 = figure;

  %
  % Plot experimental data for 4340 steel Rc 45 (Chi et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load SigEpsEp0001173KRc45.dat
  epEx = SigEpsEp0001173KRc45(:,2);
  seqEx = SigEpsEp0001173KRc45(:,3);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'ro-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'ro-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'ro--', 'LineWidth', 2); hold on;

  load SigEpsEp0001298KRc45.dat
  epEx = SigEpsEp0001298KRc45(:,2);
  seqEx = SigEpsEp0001298KRc45(:,3);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'go-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'go-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'go--', 'LineWidth', 2); hold on;
  
  load SigEpsEp0001373KRc45.dat
  epEx = SigEpsEp0001373KRc45(:,2);
  seqEx = SigEpsEp0001373KRc45(:,3);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'bo-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'bo-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'bo--', 'LineWidth', 2); hold on;
  
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
  load SigEpsEp1000173KRc45.dat
  epEx = SigEpsEp1000173KRc45(:,2);
  seqEx = SigEpsEp1000173KRc45(:,3);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'ro-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'ro-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'ro--', 'LineWidth', 2); hold on;
  
  load SigEpsEp1000298KRc45.dat
  epEx = SigEpsEp1000298KRc45(:,2);
  seqEx = SigEpsEp1000298KRc45(:,3);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'go-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'go-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'go-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'go-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'go--', 'LineWidth', 2); hold on;
  
  load SigEpsEp1000373KRc45.dat
  epEx = SigEpsEp1000373KRc45(:,2);
  seqEx = SigEpsEp1000373KRc45(:,3);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  Rc = 45.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  figure(fig01);
  psigeiso = plot(eplas, sig_e/1.0e6, 'bo-'); hold on;
  set(psigeiso, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'bo-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'bo-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'bo--', 'LineWidth', 2); hold on;
  
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

  %
  % Plot experimental data for 4340 steel Rc 49 (Chi2 et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load SigEpsEp0001173KRc49.dat
  epEx = SigEpsEp0001173KRc49(:,2);
  seqEx = SigEpsEp0001173KRc49(:,3);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso1 = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso1, 'LineWidth', 3);
  figure(fig01);
  psigeiso1 = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso1, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'ro-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'ro-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'ro--', 'LineWidth', 2); hold on;

  load SigEpsEp0001298KRc49.dat
  epEx = SigEpsEp0001298KRc49(:,2);
  seqEx = SigEpsEp0001298KRc49(:,3);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso2 = plot(sigma_e/1.0e6, theta/1.0e6, 'go--'); hold on;
  set(ptangiso2, 'LineWidth', 3);
  figure(fig01);
  psigeiso2 = plot(eplas, sig_e/1.0e6, 'go--'); hold on;
  set(psigeiso2, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'go-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'go-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'go--', 'LineWidth', 2); hold on;
  
  load SigEpsEp0001373KRc49.dat
  epEx = SigEpsEp0001373KRc49(:,2);
  seqEx = SigEpsEp0001373KRc49(:,3);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso3 = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-.'); hold on;
  set(ptangiso3, 'LineWidth', 3);
  figure(fig01);
  psigeiso3 = plot(eplas, sig_e/1.0e6, 'bo-.'); hold on;
  set(psigeiso3, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'bo-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'bo-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6, 'bo--', 'LineWidth', 2); hold on;
  
  figure(fig00);
  set(gca, 'XLim', [0 500], 'YLim', [0 25000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;
  legend([ptangiso1 ptangiso2 ptangiso3],'T = 173 K','T = 298 K','T = 373 K');

  figure(fig01);
  set(gca, 'XLim', [0 0.25], 'YLim', [0 450] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;
  legend([psigeiso1 psigeiso2 psigeiso3],'T = 173 K','T = 298 K','T = 373 K');

%====================================================================

function plotTangChiRc49_1000

  fig00 = figure;
  fig01 = figure;
  fig02 = figure;

  %
  % Plot experimental data for 4340 steel Rc 49 (Chi2 et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load SigEpsEp1000173KRc49.dat
  epEx = SigEpsEp1000173KRc49(:,2);
  seqEx = SigEpsEp1000173KRc49(:,3);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso1 = plot(sigma_e/1.0e6, theta/1.0e6, 'ro-'); hold on;
  set(ptangiso1, 'LineWidth', 3);
  figure(fig01);
  psigeiso1 = plot(eplas, sig_e/1.0e6, 'ro-'); hold on;
  set(psigeiso1, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  sig_a = 50.0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'ro-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'ro-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6 - sig_a, 'ro--', 'LineWidth', 2); hold on;
  
  load SigEpsEp1000298KRc49.dat
  epEx = SigEpsEp1000298KRc49(:,2);
  seqEx = SigEpsEp1000298KRc49(:,3);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso2 = plot(sigma_e/1.0e6, theta/1.0e6, 'go--'); hold on;
  set(ptangiso2, 'LineWidth', 3);
  figure(fig01);
  psigeiso2 = plot(eplas, sig_e/1.0e6, 'go--'); hold on;
  set(psigeiso2, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'go-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'go-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6 - sig_a, 'go--', 'LineWidth', 2); hold on;
  
  load SigEpsEp1000373KRc49.dat
  epEx = SigEpsEp1000373KRc49(:,2);
  seqEx = SigEpsEp1000373KRc49(:,3);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  Rc = 49.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  figure(fig00);
  ptangiso3 = plot(sigma_e/1.0e6, theta/1.0e6, 'bo-.'); hold on;
  set(ptangiso3, 'LineWidth', 3);
  figure(fig01);
  psigeiso3 = plot(eplas, sig_e/1.0e6, 'bo-.'); hold on;
  set(psigeiso3, 'LineWidth', 3);
  figure(fig02);
  mu_mu0 = mu/mu0;
  psigi = plot(eplas, mu_mu0.*sig_i/1.0e6, 'bo-.', 'LineWidth', 2); hold on;
  psige = plot(eplas, mu_mu0.*sig_e/1.0e6, 'bo-', 'LineWidth', 2); hold on;
  psigeps = plot(epEx, seqEx/1.0e6 - sig_a, 'bo--', 'LineWidth', 2); hold on;
  
  figure(fig00);
  set(gca, 'XLim', [0 400], 'YLim', [0 30000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\sigma_e (MPa)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;
  legend([ptangiso1 ptangiso2 ptangiso3],'T = 173 K','T = 298 K','T = 373 K');

  figure(fig01);
  set(gca, 'XLim', [0 0.07], 'YLim', [0 350] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('\epsilon_p', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\sigma_e (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;
  legend([psigeiso1 psigeiso2 psigeiso3],'T = 173 K','T = 298 K','T = 373 K');
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

%  [n, m] = size(ep);
%  count = 1;
%  for i=1:n
%    if ~(ep(i) < 0.0)
%      sig_y(count) = sig(i);
%      sig_m = sig(i);
%      [sig_e(count), sig_i(count),mu0(count),mu(count)] = ...
%         computeSige(sig_m, epdot, T0, T0, Rc);
%      eplas(count) = ep(i);
%      count = count+1;
%    end
%  end
%  n = length(eplas);
%  count = 1;
%  for i=1:n-1
%    dep = eplas(i+1) - eplas(i);
%    dsig = sig_y(i+1) - sig_y(i);
%    theta(count) = dsig/dep;
%    sigma_y(count) = 0.5*(sig_y(i+1) + sig_y(i));
%    sigma_e(count) = 0.5*(sig_e(i+1) + sig_e(i));
%    count = count + 1;
%  end
%  sigma_e = sigma_y;
%  sig_e = sig_y;

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
  %sig_e = sig_e - sig_e(1);
  n = length(eplas);
  count = 1;
  for i=1:n-1
    dep = eplas(i+1) - eplas(i);
    dsig = sig_e(i+1) - sig_e(i);
    theta(count) = dsig/dep;
    %sigma_e(count) = 0.5*(sig_e(i+1) + sig_e(i));
    sigma_e(count) = sig_e(i+1);
    count = count + 1;
  end

%
% Compute the tangent modulus and sigma_e (adiabatic)
%
function [theta, sigma_e, eplas, sig_e, sig_i, mu0, mu] = ...
          computeTangentModulusAdi(delT, ep, sig, epdot, T0, Rc)

%  [n, m] = size(ep);
%  count = 1;
%  for i=1:n
%    if ~(ep(i) < 0.0)
%      sig_y(count) = sig(i);
%      sig_m = sig(i);
%      ep_m =  ep(i);
%      T = computeTemp(delT, sig_m, ep_m, epdot, T0);
%      [sig_e(count), sig_i(count),mu0(count),mu(count)] = ...
%         computeSige(sig_m, epdot, T, T0, Rc);
%      eplas(count) = ep(i);
%      count = count+1;
%    end
%  end
%  n = length(eplas);
%  count = 1;
%  for i=1:n-1
%    dep = eplas(i+1) - eplas(i);
%    dsig = sig_y(i+1) - sig_y(i);
%    theta(count) = dsig/dep;
%    sigma_y(count) = 0.5*(sig_y(i+1) + sig_y(i));
%    sigma_e(count) = 0.5*(sig_e(i+1) + sig_e(i));
%    count = count + 1;
%  end
%  sigma_e = sigma_y;
%  sig_e = sig_y;

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
  %sig_e = sig_e - sig_e(1);
  n = length(eplas);
  count = 1;
  for i=1:n-1
    dep = eplas(i+1) - eplas(i);
    dsig = sig_e(i+1) - sig_e(i);
    if (dep ~= 0.0)
      theta(count) = dsig/dep;
      %sigma_e(count) = 0.5*(sig_e(i+1) + sig_e(i));
      sigma_e(count) = sig_e(i+1);
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

  %
  % Best fit
  %
  %edot_0i = 1.0e8;
  %p_i = 2.0/3.0;
  %q_i = 1.0;
  %g_0i = -1.5425e-3*Rc^3 + 2.0396e-1*Rc^2 - 8.8536*Rc + 1.27e2;
  %sigma_i = 0.18162*Rc^3 - 24.029*Rc^2 + 1077.1*Rc - 14721;

  %
  % Multiple temperature fit
  %
  edot_0i = 1.0e8;
  p_i = 2.0/3.0;
  q_i = 1.0;
  if (T < 1040)
    FisherRawData = [[30   8.6760e+08   3.3102e+00];...
                     [38   1.5287e+09   4.1211e-01];...
                     [45   1.6511e+09   1.0482e+00];...
                     [49   1.7705e+09   1.2546e+00]];
    if (Rc == 30)
      sigma_i = 8.6760e+02;
      g_0i = 3.3102e+00;
    elseif (Rc == 38)
      sigma_i = 1.5287e+03;
      g_0i = 4.1211e-01;
    elseif (Rc == 45)
      sigma_i = 1.6511e+03;
      g_0i = 1.0482e+00;
    else
      sigma_i = 1.7705e+03;
      g_0i = 1.2546e+00;
    end
    %g_0i    = -1.7778e-03*Rc^3 + 2.3110e-01*Rc^2 - 9.8832e+00*Rc + 1.3982e+02;
    %sigma_i =  0.2887*Rc^3 - 36.854*Rc^2 + 1586.3*Rc - 21322;
  else
    g_0i = 0.57563;
    sigma_i = 896.32;
  end
  sigma_i = sigma_i*1.0e6;

  %
  % HY-100 fit
  %
  %edot_0i = 1.0e13;
  %p_i = 0.5;
  %q_i = 1.5;
  %g_0i = -2.0898e-2*Rc^3 + 2.7668*Rc^2 - 12.096*Rc + 1.7494e3;
  %sigma_i = 0.6044*Rc^3 - 7.7094e1*Rc^2 + 3.2569e3*Rc - 4.3769e4;

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
  S_e = (1.0 - (kappa*T/(mu*b^3*g_0e)*log(edot_0e/epdot))^(1/q_e))^(1/p_e);

  %
  % Compute sig_e
  %
  %sigma_e = (1.0/S_e)*(mu_0/mu*(sig_y - sig_a) - S_i*sigma_i);
  sigma_e = (mu_0/mu*(sig_y - sig_a) - S_i*sigma_i);
  
  %[sigma_e mu_0 mu sig_y sig_a S_i sigma_i]
  sigma_i = S_i*sigma_i;
