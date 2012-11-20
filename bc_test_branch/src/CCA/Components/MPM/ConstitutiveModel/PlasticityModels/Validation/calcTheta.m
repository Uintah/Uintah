function calcTheta

  %
  % Plot the tangent modulus vs sigma_e to find sigma_es
  %
  plotTangentModulus;

%
% Plot the tangent modulus vs sigma_e to find sigma_es
%
function plotTangentModulus

  [theta30_1] = plotTangJCRc30Tension;
  [theta30_2] = plotTangJCRc30Shear;
  theta30 = cat(1,theta30_1,theta30_2);
  [theta0fit30, thetaIVfit30] = plotThetavsEdot(theta30, '4340 Steel Rc 30');
  [a0_30, aIV_30] = plotThetavsT(theta30, theta0fit30, thetaIVfit30, '4340 Steel Rc 30');

  [theta38_1] = plotTangLarsonRc38;
  [theta38_2] = plotTangLYRc38500;
  [theta38_3] = plotTangLYRc381500;
  [theta38_4] = plotTangLYRc382500;
  theta38 = cat(1,theta38_1,theta38_2,theta38_3,theta38_4);
  [theta0fit38, thetaIVfit38] = plotThetavsEdot(theta38, '4340 Steel Rc 38');
  [a0_38, aIV_38] = plotThetavsT(theta38, theta0fit38, thetaIVfit38,  '4340 Steel Rc 38');

  [theta45_1] = plotTangChiRc45_0001;
  [theta45_2] = plotTangChiRc45_1000;
  theta45 = cat(1,theta45_1,theta45_2);
  [theta0fit45, thetaIVfit45] = plotThetavsEdot(theta45, '4340 Steel Rc 45');
  [a0_45, aIV_45] = plotThetavsT(theta45, theta0fit45, thetaIVfit45, '4340 Steel Rc 45');

  [theta49_1] = plotTangChiRc49_0001;
  [theta49_2] = plotTangChiRc49_1000;
  theta49 = cat(1,theta49_1,theta49_2);
  [theta0fit49, thetaIVfit49] = plotThetavsEdot(theta49, '4340 Steel Rc 49');
  [a0_49, aIV_49] = plotThetavsT(theta49, theta0fit49, thetaIVfit49, '4340 Steel Rc 49');

  Rc = [30 38 45 49];
  a0_0 = [a0_30(1) a0_38(1) a0_45(1) a0_49(1)];
  a0_1 = [a0_30(2) a0_38(2) a0_45(2) a0_49(2)];
  a0_2 = [a0_30(3) a0_38(3) a0_45(3) a0_49(3)];
  a0_3 = [a0_30(4) a0_38(4) a0_45(4) a0_49(4)];
  aIV_0 = [aIV_30(1) aIV_38(1) aIV_45(1) aIV_49(1)];
  aIV_1 = [aIV_30(2) aIV_38(2) aIV_45(2) aIV_49(2)];
  aIV_2 = [aIV_30(3) aIV_38(3) aIV_45(3) aIV_49(3)];
  aIV_3 = [aIV_30(4) aIV_38(4) aIV_45(4) aIV_49(4)];

  Rcmin = 25;
  Rcmax = 50;
  nRc = 100;
  dRc = (Rcmax - Rcmin)/nRc;
  for i=1:nRc+1
    RcFit(i) = Rcmin + (i-1)*dRc;
  end
  [fita0_0] = polyfit(Rc, a0_0, 3)
  [a0_0Fit] = polyval(fita0_0, RcFit);
  [fita0_1] = polyfit(Rc, a0_1, 3)
  [a0_1Fit] = polyval(fita0_1, RcFit);
  [fita0_2] = polyfit(Rc, a0_2, 3)
  [a0_2Fit] = polyval(fita0_2, RcFit);
  [fita0_3] = polyfit(Rc, a0_3, 3)
  [a0_3Fit] = polyval(fita0_3, RcFit);

  [fitaIV_0] = polyfit(Rc, aIV_0, 3)
  [aIV_0Fit] = polyval(fitaIV_0, RcFit);
  [fitaIV_1] = polyfit(Rc, aIV_1, 3)
  [aIV_1Fit] = polyval(fitaIV_1, RcFit);
  [fitaIV_2] = polyfit(Rc, aIV_2, 3)
  [aIV_2Fit] = polyval(fitaIV_2, RcFit);
  [fitaIV_3] = polyfit(Rc, aIV_3, 3)
  [aIV_3Fit] = polyval(fitaIV_3, RcFit);

  fig00 = figure;
  subplot(2,2,1);
  plot(Rc, a0_0, 'ro'); hold on;
  plot(RcFit, a0_0Fit, 'g-');
  subplot(2,2,2);
  plot(Rc, a0_1, 'rs'); hold on;
  plot(RcFit, a0_1Fit, 'g-');
  subplot(2,2,3);
  plot(Rc, a0_2, 'rd'); hold on;
  plot(RcFit, a0_2Fit, 'g-');
  subplot(2,2,4);
  plot(Rc, a0_3, 'rp'); hold on;
  plot(RcFit, a0_3Fit, 'g-');

  fig01 = figure;
  subplot(2,2,1);
  plot(Rc, aIV_0, 'bo'); hold on;
  plot(RcFit, aIV_0Fit, 'g-');
  subplot(2,2,2);
  plot(Rc, aIV_1, 'bs'); hold on;
  plot(RcFit, aIV_1Fit, 'g-');
  subplot(2,2,3);
  plot(Rc, aIV_2, 'bd'); hold on;
  plot(RcFit, aIV_2Fit, 'g-');
  subplot(2,2,4);
  plot(Rc, aIV_3, 'bp'); hold on;
  plot(RcFit, aIV_3Fit, 'g-');

%====================================================================
function [fit0WRTedot, fitIVWRTedot] = plotThetavsEdot(theta, label)

  format short e;
  theta = sortrows(theta, 3);
  theta0 = theta(:,2);
  thetaIV = theta(:,1)+theta(:,2);
  edot = theta(:,3);
  T = theta(:,4);

  Tu = unique(T);
  for i=1:length(Tu)
    count = 1;
    for j=1:length(T)
      if (Tu(i) == T(j))
        thetaEdot(i,count,1) = theta0(j);
        thetaEdot(i,count,2) = thetaIV(j);
        thetaEdot(i,count,3) = edot(j);
        thetaEdot(i,count,4) = T(j);
        count = count+1;
      end
    end
  end

  for kk=1:length(Tu)
    theta0_val = thetaEdot(kk,:,1);
    thetaIV_val = thetaEdot(kk,:,2);
    edot_val = thetaEdot(kk,:,3);
    T_val = thetaEdot(kk,:,4);
    n = length(edot_val);
    count = 1;
    for j=1:n
      if (edot_val(j) ~= 0)
        t0_val(count) = theta0_val(j);
        tIV_val(count) = thetaIV_val(j);
        ed_val(count) = edot_val(j);
        tt_val(count) = T_val(j);
        count = count+1;
      end
    end
    data_val = cat(1,t0_val,tIV_val,ed_val,tt_val);

    if (length(ed_val) < 2) 
      continue;
    end

    %
    % Fit a polygon to sqrt(edot) data
    %
    pfit0 = polyfit(sqrt(ed_val), t0_val, 1);
    pfitIV = polyfit(sqrt(ed_val), tIV_val, 1);
    edotmax = max(sqrt(ed_val));
    edotmin = min(sqrt(ed_val));
    nedot = 100;
    dedot = (edotmax-edotmin)/nedot;
    for i=1:nedot+1
      edotfit(i) = edotmin + (i-1)*dedot;
    end
    theta0Fit = polyval(pfit0, edotfit);
    thetaIVFit = polyval(pfitIV, edotfit);

    %figure;
    %subplot(2,2,1);
    %semilogx(ed_val, t0_val, 'ro-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,2);
    %semilogx(ed_val, tIV_val, 'bs-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,3);
    %plot(sqrt(ed_val), t0_val, 'ro-', 'LineWidth', 3);  hold on;
    %plot(edotfit, theta0Fit, 'r--', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot^{1/2} (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,4);
    %plot(sqrt(ed_val), tIV_val, 'bs-', 'LineWidth', 3);  hold on;
    %plot(edotfit, thetaIVFit, 'b--', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot^{1/2} (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;

    %
    % Subtract the fit parameters from theta0
    %
    a1 = pfit0(1);
    a0 = pfit0(2);
    fit0(1) = a0;
    fit0(2) = a1;
    tt0 = a0 + a1*sqrt(ed_val);
    
    a1 = pfitIV(1);
    a0 = pfitIV(2);
    fitIV(1) = a0;
    fitIV(2) = a1;
    ttIV = a0 + a1*sqrt(ed_val);

    t0_val = t0_val - tt0;
    tIV_val = tIV_val - ttIV;


    %
    % Fit a polygon to log(edot) data
    %
    pfit0 = polyfit(log(ed_val), t0_val, 1);
    pfitIV = polyfit(log(ed_val), tIV_val, 1);
    edotmax = max(ed_val);
    edotmin = min(ed_val);
    nedot = 100;
    dedot = (edotmax-edotmin)/nedot;
    for i=1:nedot+1
      edotfit(i) = edotmin + (i-1)*dedot;
    end
    theta0Fit = polyval(pfit0, log(edotfit));
    thetaIVFit = polyval(pfitIV, log(edotfit));

    %figure;
    %subplot(2,2,1);
    %semilogx(ed_val, t0_val, 'ro-', 'LineWidth', 3);  hold on;
    %semilogx(edotfit, theta0Fit, 'r--', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,2);
    %semilogx(ed_val, tIV_val, 'bs-', 'LineWidth', 3);  hold on;
    %semilogx(edotfit, theta0Fit, 'b--', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,3);
    %plot(sqrt(ed_val), t0_val, 'ro-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot^{1/2} (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,4);
    %plot(sqrt(ed_val), tIV_val, 'bs-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot^{1/2} (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
  
    %
    % Subtract the fit parameters from theta0
    %
    a1 = pfit0(1);
    a0 = pfit0(2);
    fit0(3) = a0;
    fit0(4) = a1;
    tt0 = a0 + a1*log(ed_val);
    
    a1 = pfitIV(1);
    a0 = pfitIV(2);
    fitIV(3) = a0;
    fitIV(4) = a1;
    ttIV = a0 + a1*log(ed_val);

    t0_val = t0_val - tt0;
    tIV_val = tIV_val - ttIV;

    %figure;
    %subplot(2,2,1);
    %semilogx(ed_val, t0_val, 'ro-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,2);
    %semilogx(ed_val, tIV_val, 'bs-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,3);
    %plot(sqrt(ed_val), t0_val, 'ro-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot^{1/2} (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;
    %subplot(2,2,4);
    %plot(sqrt(ed_val), tIV_val, 'bs-', 'LineWidth', 3);  hold on;
    %set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
    %xlabel('edot^{1/2} (/s)', 'FontName', 'bookman', 'FontSize', 16);
    %ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
    %title(label, 'FontName', 'bookman', 'FontSize', 16);
    %grid on;

    %
    % Finally plot the total fit curve
    %
    for i=1:length(edotfit)
      theta0_fitted(i) = fit0(1) + fit0(3) + fit0(2)*sqrt(edotfit(i)) + ...
                         fit0(4)*log(edotfit(i));
      thetaIV_fitted(i) = fitIV(1) + fitIV(3) + fitIV(2)*sqrt(edotfit(i)) + ...
                          fitIV(4)*log(edotfit(i));
    end
 
    figure;
    str = sprintf('%s T = %g', label, T_val(1))
    str0 = sprintf('%g + %g log(edot) + %g sqrt(edot)', (fit0(1)+fit0(3)), ...
           fit0(4), fit0(2));
    strIV = sprintf('%g + %g log(edot) + %g sqrt(edot)', (fitIV(1)+fitIV(3)), ...
           fitIV(4), fitIV(2));
    subplot(1,2,1);
    semilogx(edot_val, theta0_val, 'ro-', 'LineWidth', 3); hold on;
    semilogx(edotfit, theta0_fitted, 'g--', 'LineWidth', 3); hold on;
    gg = gtext(str0);
    set(gg, 'Rotation', 90);
    title(str);
    subplot(1,2,2);
    semilogx(edot_val, thetaIV_val, 'mo-', 'LineWidth', 3); hold on;
    semilogx(edotfit, thetaIV_fitted, 'g--', 'LineWidth', 3); hold on;
    gg = gtext(strIV);
    set(gg, 'Rotation', 90);
    title(str);

    %
    % Save the fit data
    %
    theta0FitData(kk,:) = [T_val(1) (fit0(1)+fit0(3)) fit0(4) fit0(2)];
    thetaIVFitData(kk,:) = [T_val(1) (fitIV(1)+fitIV(3)) fitIV(4) fitIV(2)];

    clear theta0_val thetaIV_val edot_val T_val;
    clear t0_val tIV_val ed_val tt_val;
    clear data_val;
  end

  %
  % Average the fit data (temperature independent)
  %
  fit0WRTedot = mean(theta0FitData,1);
  fitIVWRTedot = mean(thetaIVFitData,1);

%====================================================================
function [a0, aIV] = plotThetavsT(theta, Efit0, EfitIV, label)

  %
  % Initial data
  %
  theta0 = theta(:,2);
  thetaIV = theta(:,1)+theta(:,2);
  edot = theta(:,3);
  T = theta(:,4);

  %
  % remove edot dependence
  %
  theta0 = theta0 - Efit0(2) - Efit0(3)*log(edot) - Efit0(4)*sqrt(edot);
  thetaIV = thetaIV - EfitIV(2) - EfitIV(3)*log(edot) - EfitIV(4)*sqrt(edot);

  Tmax = max(T);
  Tmin = min(T);
  nT = 100;
  dT = (Tmax-Tmin)/nT;
  for i=1:nT+1
    Tfit(i) = Tmin + (i-1)*dT;
  end

  [Tfit0] = polyfit(T, theta0, 1);
  [TfitIV] = polyfit(T, thetaIV, 1);
  theta0Tfit = polyval(Tfit0, Tfit);
  thetaIVTfit = polyval(TfitIV, Tfit);

  str0 = sprintf('\\theta_0 = %g + %g log(edot) + %g sqrt(edot) + %g T', ...
        (Efit0(2)+Tfit0(2)),Efit0(3),Efit0(4),Tfit0(1));
  strIV = sprintf('\\theta_IV = %g + %g log(edot) + %g sqrt(edot) + %g T', ...
        (EfitIV(2)+TfitIV(2)),EfitIV(3),EfitIV(4),TfitIV(1));

  figure;
  subplot(1,2,1);
  plot(T, theta0, 'ro-', 'LineWidth', 3);  hold on;
  plot(Tfit, theta0Tfit, 'r--', 'LineWidth', 2);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('T (K)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta_0 (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title(label, 'FontName', 'bookman', 'FontSize', 16);
  gg = gtext(str0);
  set(gg, 'Rotation', 90);
  grid on;
  subplot(1,2,2);
  plot(T, thetaIV, 'bs-', 'LineWidth', 3);  hold on;
  plot(Tfit, thetaIVTfit, 'b--', 'LineWidth', 2);
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('T (K)', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta_{IV} (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title(label, 'FontName', 'bookman', 'FontSize', 16);
  gg = gtext(strIV);
  set(gg, 'Rotation', 90);
  grid on;

  a0(1) = Efit0(2) + Tfit0(2);
  a0(2) = Efit0(3);
  a0(3) = Efit0(4);
  a0(4) = Tfit0(1);
  aIV(1) = EfitIV(2) + TfitIV(2);
  aIV(2) = EfitIV(3);
  aIV(3) = EfitIV(4);
  aIV(4) = TfitIV(1);

%====================================================================
function [theta0] = plotTangJCRc30Tension

  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

  E = 213.0e9;

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'g^-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);
  
  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'bp-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);
  
  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];

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
  Rc = 30;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'md-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'm--');
  set(plfit, 'LineWidth', 2);

  theta0(4,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 8000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 30 (Tension)', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;

function [theta0] = plotTangJCRc30Shear

  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

  E = 213.0e9;

  %
  % Load experimental data from Johnson-Cook (Rc = 30)
  %

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'rx-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);
  
  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'gs-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);
  
  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'bv-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);
 
  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];

  %set(gca, 'XLim', [0 1], 'YLim', [0 8000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 30 (Shear)', 'FontName', 'bookman', 'FontSize', 16);
  grid on;
  axis square;

  %====================================================================

function [theta0] = plotTangLarsonRc38

  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'gx-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38.0;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'bs-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 5000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 0.0002/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  %====================================================================

function [theta0] = plotTangLYRc38500

  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'gx-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'bs-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 16000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 500/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  %====================================================================

function [theta0] = plotTangLYRc381500

  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'gx-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'bs-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'mv-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  theta0(4,:) = [pfit(1) pfit(2) epdot T Rc];

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'm--');
  set(plfit, 'LineWidth', 2);
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'kd-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'k--');
  set(plfit, 'LineWidth', 2);

  theta0(5,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 8000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 1500/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square

  %====================================================================
         
function [theta0] = plotTangLYRc382500

  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'ro-'); hold on;
  set(ptangiso, 'LineWidth', 3);
  
  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];

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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'gx-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'bs-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  Rc = 38;
  seqEx = seqEx*1.0e6;
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
    computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'mv-'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'm--');
  set(plfit, 'LineWidth', 2);

  theta0(4,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 4000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 38 2500/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  %====================================================================
         
function [theta0] = plotTangChiRc45_0001

  E = 213.0e9;
  fig00 = figure;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'r-o'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'g-x'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'b-s'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 15000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45 0.0001/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  %====================================================================

function [theta0] = plotTangChiRc45_1000

  fig00 = figure;
  E = 213.0e9;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'r-o'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'g-x'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'b-s'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 20000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45 1000/s', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

  %====================================================================

function [theta0] = plotTangChiRc49_0001

  fig00 = figure;

  E = 213.0e9;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'r-o'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'g-x'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusIso(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'b-s'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 30000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49', 'FontName', 'bookman', 'FontSize', 16);
  axis square;

function [theta0] = plotTangChiRc49_1000

  fig00 = figure;

  E = 213.0e9;

  sigemin = 0.0;
  sigemax = 1.0;
  nsige = 100;
  dsige = (sigemax-sigemin)/nsige;
  for i=1:nsige+1 
    sigefit(i) = sigemin + (i-1)*dsige;
  end

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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'r-o'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'r--');
  set(plfit, 'LineWidth', 2);

  theta0(1,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'g-x'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'g--');
  set(plfit, 'LineWidth', 2);

  theta0(2,:) = [pfit(1) pfit(2) epdot T Rc];
  
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
  [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
     computeTangentModulusAdi(delT, epEx, seqEx, epdot, T, Rc);
  theta = theta*1.0e-6;
  ptangiso = plot(sigma_e, theta, 'b-s'); hold on;
  set(ptangiso, 'LineWidth', 3);

  pfit = polyfit(sigma_e, theta, 1); 
  thetafit = polyval(pfit, sigefit);
  plfit = plot(sigefit, thetafit, 'b--');
  set(plfit, 'LineWidth', 2);

  theta0(3,:) = [pfit(1) pfit(2) epdot T Rc];
  
  %set(gca, 'XLim', [0 1], 'YLim', [0 30000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('tanh(\alpha \sigma_e/\sigma_{es})/tanh(\alpha) ', ...
         'FontName', 'bookman', 'FontSize', 16);
  ylabel('\theta (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49', 'FontName', 'bookman', 'FontSize', 16);
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
function [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
          computeTangentModulusIso(delT, ep, sig, epdot, T0, Rc)

  [n, m] = size(ep);
  count = 1;
  for i=1:n
    if ~(ep(i) < 0.0)
      sig_m = sig(i);
      [sig_es(count), sig_e(count), sig_i(count),mu0(count),mu(count)] = ...
         computeSige(sig_m, epdot, T0, T0, Rc);
      eplas(count) = ep(i);
      count = count+1;
    end
  end
  %T0
  sig_e = sig_e - sig_e(1);
  n = length(eplas);
  count = 1;
  for i=1:n-1
    dep = eplas(i+1) - eplas(i);
    dsig = sig_e(i+1) - sig_e(i);
    dsigdep = dsig/dep;
    %if (dsigdep > 0.0) 
      theta(count) = dsigdep;
      sigma_e(count) = sig_e(i)/sig_es(i);
      alpha = 3;
      sigma_e(count) = tanh(alpha*sigma_e(count))/tanh(alpha);
      count = count + 1;
    %end
  end
  %theta(count) = theta(count-1);
  %sigma_e(count) = sig_e(n)/sig_es(n);
  %sigma_e(count) = tanh(alpha*sigma_e(count))/tanh(alpha);

%
% Compute the tangent modulus and sigma_e (adiabatic)
%
function [theta, sigma_e, eplas, sig_es, sig_e, sig_i, mu0, mu] = ...
          computeTangentModulusAdi(delT, ep, sig, epdot, T0, Rc)

  [n, m] = size(ep);
  count = 1;
  for i=1:n
    if ~(ep(i) < 0.0)
      sig_m = sig(i);
      ep_m =  ep(i);
      T = computeTemp(delT, sig_m, ep_m, epdot, T0);
      [sig_es(count), sig_e(count), sig_i(count),mu0(count),mu(count)] = ...
         computeSige(sig_m, epdot, T, T0, Rc);
      eplas(count) = ep(i);
      count = count+1;
    end
  end
  %T
  sig_e = sig_e - sig_e(1);
  n = length(eplas);
  count = 1;
  for i=1:n-1
    dep = eplas(i+1) - eplas(i);
    dsig = sig_e(i+1) - sig_e(i);
    dsigdep = dsig/dep;
    %if (dsigdep > 0.0) 
      theta(count) = dsigdep;
      sigma_e(count) = sig_e(i)/sig_es(i);
      alpha = 3;
      sigma_e(count) = tanh(alpha*sigma_e(count))/tanh(alpha);
      count = count + 1;
    %end
  end
  %theta(count) = theta(count-1);
  %sigma_e(count) = sig_e(n)/sig_es(n);
  %sigma_e(count) = tanh(alpha*sigma_e(count))/tanh(alpha);

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
function [sigma_es, sigma_e, sigma_i, mu_0, mu] = computeSige(sig_y, epdot, T, T0, Rc)

  %
  % Constants
  %
  kappa = 1.3806503e-23;
  b = 2.48e-10;

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
  % Compute sigma_es
  %
  g_0es = 5.031e-4*Rc^3 - 5.74e-2*Rc^2 + 2.1451*Rc - 26.054;
  sigma_es0 = 0.70417*Rc^3  - 85.561*Rc^2 + 3377.3*Rc - 42583.0;
  %g_0es = 5.85e-3*Rc - 8.92e-3;
  %sigma_es0 = -28.1*Rc + 1688.0;
  sigma_es0 = sigma_es0*1.0e6;
  edot_0es = 1.0e7;
  sigma_es = sigma_es0*(epdot/edot_0es)^(kappa*T/(mu*b^3*g_0es)); 

  %
  % Compute S_i
  %
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
