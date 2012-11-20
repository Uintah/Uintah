function testModelSt

  fid = fopen('StPickData.dat', 'w');
  %
  % Plot experimental data for 4340 steel Rc 49 (Chi2 et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  fig00 = figure;
  load FlowSt0001173KChi2.dat
  epsEx = FlowSt0001173KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001173KChi2(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp00001173 = plot(epsEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp00001173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001298KChi2.dat
  epsEx = FlowSt0001298KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001298KChi2(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp00001298 = plot(epsEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp00001298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001373KChi2.dat
  epsEx = FlowSt0001373KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001373KChi2(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp00001373 = plot(epsEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp00001373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  set(gca, 'XLim', [0 0.25], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp00001173 pexp00001298 pexp00001373], ...
         '0.0001/s 173 K Chi(1989)', ...
         '0.0001/s 298 K Chi(1989)', ...
         '0.0001/s 373 K Chi(1989)');
  axis square;

  %====================================================================

  fig01 = figure;

  %
  % Plot experimental data for 4340 steel Rc 45 (Chi et al)
  % (data in the form of shear stress vs shear strain)
  % (quasistatic)
  %
  load FlowSt0001173KChi.dat
  epsEx = FlowSt0001173KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001173KChi(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp00001173 = plot(epsEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp00001173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001298KChi.dat
  epsEx = FlowSt0001298KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001298KChi(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp00001298 = plot(epsEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp00001298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001373KChi.dat
  epsEx = FlowSt0001373KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001373KChi(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp00001373 = plot(epsEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp00001373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  set(gca, 'XLim', [0 0.3], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp00001173 pexp00001298 pexp00001373], ...
         '0.0001/s 173 K Chi(1989)', ...
         '0.0001/s 298 K Chi(1989)', ...
         '0.0001/s 373 K Chi(1989)');
  axis square;

  %====================================================================

  fig10 = figure;

  %
  % Plot experimental data for 4340 steel Rc 49 (Chi2 et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load FlowSt1000173KChi2.dat
  epsEx = FlowSt1000173KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000173KChi2(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp1000173 = plot(epsEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp1000173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000298KChi2.dat
  epsEx = FlowSt1000298KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000298KChi2(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp1000298 = plot(epsEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp1000298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000373KChi2.dat
  epsEx = FlowSt1000373KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000373KChi2(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp1000373 = plot(epsEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp1000373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  set(gca, 'XLim', [0 0.1], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 49', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp1000173 pexp1000298 pexp1000373], ...
         '1000/s 173 K Chi(1989)', ...
         '1000/s 298 K Chi(1989)', ...
         '1000/s 373 K Chi(1989)');
  axis square;

  %====================================================================

  fig11 = figure;

  %
  % Plot experimental data for 4340 steel Rc 45 (Chi et al)
  % (data in the form of shear stress vs shear strain)
  % (dynamic)
  %
  load FlowSt1000173KChi.dat
  epsEx = FlowSt1000173KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000173KChi(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp1000173 = plot(epsEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp1000173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000298KChi.dat
  epsEx = FlowSt1000298KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000298KChi(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp1000298 = plot(epsEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp1000298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000373KChi.dat
  epsEx = FlowSt1000373KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000373KChi(:,2)*sqrt(3);
  seqEx = seqEx.*(1.0 + epsEx);
  epsEx = log(1.0 + epsEx);
  pexp1000373 = plot(epsEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp1000373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  set(gca, 'XLim', [0 0.1], 'YLim', [0 2500] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  title('4340 Steel Rc 45', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp1000173 pexp1000298 pexp1000373], ...
         '1000/s 173 K Chi(1989)', ...
         '1000/s 298 K Chi(1989)', ...
         '1000/s 373 K Chi(1989)');
  axis square;

  %====================================================================

  fig20 = figure;
  %set(fig1, 'Position', [378 479 1147 537]);

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
  pexp00002258 = plot(epsEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp00002258,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0002;
  T = 258.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.0002/s 298 K
  %
  load FlowSt0001298KLarson.dat;
  St298K00002 = FlowSt0001298KLarson;
  epsEx = St298K00002(:,1);
  seqEx = St298K00002(:,2)*6.894657;
  pexp00002298 = plot(epsEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp00002298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 1.0 0.2]);

  delT = 10.0;
  epdot = 0.0002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.2]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.2]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.0002/s 373 K
  %
  load FlowSt0001373KLarson.dat;
  St373K00002 = FlowSt0001373KLarson;
  epsEx = St373K00002(:,1);
  seqEx = St373K00002(:,2)*6.894657;
  pexp00002373 = plot(epsEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp00002373,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.1 0.1]);

  delT = 10.0;
  epdot = 0.0002;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.1 0.1]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.1 0.1]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  set(gca, 'XLim', [0 0.8], 'YLim', [0 2000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp00002258 pexp00002298 pexp00002373], ...
         '0.0002/s 258 K Larson(1961)', ...
         '0.0002/s 298 K Larson(1961)', ...
         '0.0002/s 373 K Larson(1961)');
  axis square;

  %====================================================================

  fig30 = figure;
  %
  % Load experimental data from Johnson-Cook (Rc = 32)
  %

  %
  % 0.002/s 298K
  %
  load FlowSt0001298KJCTen.dat;
  St298K0002 = FlowSt0001298KJCTen;
  epsEx = St298K0002(:,1);
  seqEx = St298K0002(:,2);
  pexp0002298 = plot(epsEx, seqEx, '^-', 'LineWidth', 2); hold on;
  set(pexp0002298,'LineWidth',2,'MarkerSize',6,'Color',[0.2 0.5 1.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 32.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.5 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.5 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.009/s 298K
  %
  load FlowSt0009298KJCShear.dat;
  St298K0009 = FlowSt0009298KJCShear;
  epsEx = St298K0009(:,1)/sqrt(3.0);
  seqEx = St298K0009(:,2)*sqrt(3.0);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp0009298 = plot(epsEx, seqEx, 'v-', 'LineWidth', 3); hold on;
  set(pexp0009298,'LineWidth',3,'MarkerSize',6,'Color',[0.1 0.75 1.0]);

  delT = 1.0;
  epdot = 0.009;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 32.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.1 0.75 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.1 0.75 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.10/s 298K
  %
  load FlowSt010298KJCShear.dat;
  St298K01 = FlowSt010298KJCShear;
  epsEx = St298K01(:,1)/sqrt(3.0);
  seqEx = St298K01(:,2)*sqrt(3.0);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp01298 = plot(epsEx, seqEx, '<-', 'LineWidth', 3); hold on;
  set(pexp01298,'LineWidth',3,'MarkerSize',6,'Color',[0.2 0.8 0.2]);

  delT = 0.1;
  epdot = 0.1;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.8 0.2]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.8 0.2]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 1.1/s 298K
  %
  load FlowSt1_1298KJCShear.dat;
  St298K1 = FlowSt1_1298KJCShear;
  epsEx = St298K1(:,1)/sqrt(3.0);
  seqEx = St298K1(:,2)*sqrt(3.0);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp1298 = plot(epsEx, seqEx, '>-', 'LineWidth', 3); hold on;
  set(pexp1298,'LineWidth',3,'MarkerSize',6,'Color',[0.8 0.4 0.1]);

  delT = 0.01;
  epdot = 1.1;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.4 0.1]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.4 0.1]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  set(gca, 'XLim', [0 1.0], 'YLim', [0 1800] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp0002298 pexp0009298 pexp01298 pexp1298], ...
         '0.002/s 298 K (Tension) JC(1985)', ...
         '0.009/s 298 K (Shear) JC(1985)', ...
         '0.1/s 298 K (Shear) JC(1995)', ...
         '1.1/s 298 K (Shear) JC(1995)');
  axis square;

  %====================================================================

  %
  % Load experimental data from ASM hanbook vol 1
  %
  %load FlowSt001Temp.dat
  %ASM = FlowSt001Temp;

  %
  % 0.001/s 298K 923K temper
  %
  %E = 213.0e9;
  %sigY1 = ASM(5,3);
  %eps1 = 0.002 + sigY1/E;
  %sigY2 = ASM(5,2);
  %eps2 = 0.7*ASM(5,4)/100.0;
  %StASM298K0001(1,2) = sigY1*(1.0+eps1);
  %StASM298K0001(1,1) = log(1.0+eps1);
  %StASM298K0001(2,2) = sigY2*(1.0+eps2);
  %StASM298K0001(2,1) = log(1.0+eps2);
  %epsEx = StASM298K0001(:,1);
  %seqEx = StASM298K0001(:,2);
  %pexp0001298 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  %set(pexp0001298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  %delT = 10.0;
  %epdot = 0.0002;
  %T = 298.0;
  %rhomax = 7831.0;
  %epmax = max(epsEx);
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  %====================================================================

  fig40 = figure;
  %
  % Load experimental data from Aerospace Structural Metals handbook
  %
  load FlowSt001TempAero.dat
  Aero = FlowSt001TempAero;

  %
  % Convert to MPa and K
  %
  Aero(:,1) = (Aero(:,1) - 32.0)*5.0/9.0 + 273.0;
  Aero(:,2) = Aero(:,2)*6.894757;
  Aero(:,3) = Aero(:,3)*6.894757;
  Aero(:,4) = 0.9*Aero(:,4)/100.0;

  %
  % 0.002/s 298K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(1,2);
  eps1 = 0.02 + sigY1/E;
  StAero298K0002(1,1) = log(1.0+eps1);
  StAero298K0002(1,2) = sigY1;
  StAero298K0002(1,2) = sigY1*(1.0+eps1);
  sigY2 = Aero(1,3);
  eps2 =  Aero(1,4);
  StAero298K0002(2,1) = log(1.0+eps2);
  StAero298K0002(2,2) = sigY2;
  StAero298K0002(2,2) = sigY2*(1.0+eps2);
  epsEx = StAero298K0002(:,1);
  seqEx = StAero298K0002(:,2);
  pexp0002298 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 422K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(2,2);
  eps1 = 0.02 + sigY1/E;
  StAero422K0002(1,1) = log(1.0+eps1);
  StAero422K0002(1,2) = sigY1;
  StAero422K0002(1,2) = sigY1*(1.0+eps1);
  sigY2 = Aero(2,3);
  eps2 =  Aero(2,4);
  StAero422K0002(2,1) = log(1.0+eps2);
  StAero422K0002(2,2) = sigY2;
  StAero422K0002(2,2) = sigY2*(1.0+eps2);
  epsEx = StAero422K0002(:,1);
  seqEx = StAero422K0002(:,2);
  pexp0002422 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002422,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0;
  epdot = 0.002;
  T = 422.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 533K Rc 32
  %
  %E = 213.0e9;
  %sigY1 = Aero(3,2);
  %eps1 = 0.02 + sigY1/E;
  %StAero533K0002(1,1) = log(1.0+eps1);
  %StAero533K0002(1,2) = sigY1;
  %StAero533K0002(1,2) = sigY1*(1.0+eps1);
  %sigY2 = Aero(3,3);
  %eps2 =  Aero(3,4);
  %StAero533K0002(2,1) = log(1.0+eps2);
  %StAero533K0002(2,2) = sigY2;
  %StAero533K0002(2,2) = sigY2*(1.0+eps2);
  %epsEx = StAero533K0002(:,1);
  %seqEx = StAero533K0002(:,2);
  %pexp0002533 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  %set(pexp0002533,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  %delT = 1.0;
  %epdot = 0.002;
  %T = 533.0;
  %rhomax = 7831.0;
  %epmax = max(epsEx);
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  %eps = 0.05;
  %[p1] = intersectPoly(seqEx, epsEx, eps);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 589K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(4,2);
  eps1 = 0.02 + sigY1/E;
  StAero589K0002(1,1) = log(1.0+eps1);
  StAero589K0002(1,2) = sigY1;
  StAero589K0002(1,2) = sigY1*(1.0+eps1);
  sigY2 = Aero(4,3);
  eps2 =  Aero(4,4);
  StAero589K0002(2,1) = log(1.0+eps2);
  StAero589K0002(2,2) = sigY2;
  StAero589K0002(2,2) = sigY2*(1.0+eps2);
  epsEx = StAero589K0002(:,1);
  seqEx = StAero589K0002(:,2);
  pexp0002589 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002589,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 589.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 32.0;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 644K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(5,2);
  eps1 = 0.02 + sigY1/E;
  StAero644K0002(1,1) = log(1.0+eps1);
  StAero644K0002(1,2) = sigY1;
  StAero644K0002(1,2) = sigY1*(1.0+eps1);
  sigY2 = Aero(5,3);
  eps2 =  Aero(5,4);
  StAero644K0002(2,1) = log(1.0+eps2);
  StAero644K0002(2,2) = sigY2;
  StAero644K0002(2,2) = sigY2*(1.0+eps2);
  epsEx = StAero644K0002(:,1);
  seqEx = StAero644K0002(:,2);
  pexp0002644 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002644,'LineWidth',2,'MarkerSize',6,'Color',[0.2 0.6 0.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 644.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 32;
  [s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.6 0.0]);
  %[s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.6 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));

  set(gca, 'XLim', [0 0.2], 'YLim', [0 1200] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp0002298 pexp0002422 pexp0002589 pexp0002644], ...
         '0.002/s 298 K ASMH (1995)', ...
         '0.002/s 422 K ASMH (1995)', ...
         '0.002/s 589 K ASMH (1995)', ...
         '0.002/s 644 K ASMH (1995)');
  axis square;
         
  %====================================================================

  fig40 = figure;
  %
  % 570/s 298K
  %
  load FlowSt570298KJCTen.dat
  epsEx = FlowSt570298KJCTen(:,1);
  seqEx = FlowSt570298KJCTen(:,2);
  pexp570298 = plot(epsEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp570298,'LineWidth',2,'MarkerSize',6,'Color',[0.3 0.3 0.6]);

  delT = 1.0e-6;
  epdot = 570.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 32;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.3 0.3 0.6]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.3 0.3 0.6]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 604/s 500K
  %
  load FlowSt604500KJCTen.dat
  epsEx = FlowSt604500KJCTen(:,1);
  seqEx = FlowSt604500KJCTen(:,2);
  pexp604500 = plot(epsEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp604500,'LineWidth',2,'MarkerSize',6,'Color',[0.6 0.3 0.3]);

  delT = 1.0e-6;
  epdot = 604.0;
  T = 500.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 32;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.6 0.3 0.3]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.6 0.3 0.3]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 650/s 735K
  %
  load FlowSt650735KJCTen.dat
  epsEx = FlowSt650735KJCTen(:,1);
  seqEx = FlowSt650735KJCTen(:,2);
  pexp650735 = plot(epsEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp650735,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-6;
  epdot = 650.0;
  T = 735.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 32;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  set(gca, 'XLim', [0 0.2], 'YLim', [0 1400] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp570298 pexp604500 pexp650735], ...
         '570/s 298 K JC(1985)', ...
         '604/s 500 K JC(1985)', ...
         '650/s 735 K JC(1985)');
  axis square;

  %====================================================================

  fig50 = figure;
  %set(fig2, 'Position', [378 479 1147 537]);
  %
  % 500/s 298K
  %
  load FlowSt500298KLY.dat
  epsEx = FlowSt500298KLY(:,1)*1.0e-2;
  seqEx = FlowSt500298KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp500298 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 500/s 573K
  %
  load FlowSt500573KLY.dat
  epsEx = FlowSt500573KLY(:,1)*1.0e-2;
  seqEx = FlowSt500573KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp500573 = plot(epsEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 500/s 773K
  %
  load FlowSt500773KLY.dat
  epsEx = FlowSt500773KLY(:,1)*1.0e-2;
  seqEx = FlowSt500773KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp500773 = plot(epsEx, seqEx, '^-', 'LineWidth', 2); hold on;
  set(pexp500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 0.5]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 0.5]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 0.5]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  set(gca, 'XLim', [0 0.12], 'YLim', [0 1600] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp500298 pexp500573 pexp500773], ...
         '500/s 298 K LY(1997)', ...
         '500/s 573 K LY(1997)', ...
         '500/s 773 K LY(1997)');
  axis square;
         
  %====================================================================

  fig60 = figure;
  %set(fig3, 'Position', [378 479 1147 537]);
  %
  % 1500/s 298K
  %
  load FlowSt1500298KLY.dat
  epsEx = FlowSt1500298KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500298KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp1500298 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp1500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 573K
  %
  load FlowSt1500573KLY.dat
  epsEx = FlowSt1500573KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500573KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp1500573 = plot(epsEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp1500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 973K
  %
  load FlowSt1500973KLY.dat
  epsEx = FlowSt1500973KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500973KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp1500973 = plot(epsEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp1500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 1173K
  %
  load FlowSt15001173KLY.dat
  epsEx = FlowSt15001173KLY(:,1)*1.0e-2;
  seqEx = FlowSt15001173KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp15001173 = plot(epsEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp15001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 1373K
  %
  load FlowSt15001373KLY.dat
  epsEx = FlowSt15001373KLY(:,1)*1.0e-2;
  seqEx = FlowSt15001373KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp15001373 = plot(epsEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp15001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  
  set(gca, 'XLim', [0 0.24], 'YLim', [0 1800] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp1500298 pexp1500573 pexp1500973 pexp15001173 pexp15001373], ...
         '1500/s 298 K LY(1997)', ...
         '1500/s 573 K LY(1997)', ...
         '1500/s 873 K LY(1997)', ...
         '1500/s 1173 K LY(1997)', ...
         '1500/s 1373 K LY(1997)');
  axis square
         
  %====================================================================
         
  fig70 = figure;
  %set(fig4, 'Position', [378 479 1147 537]);
  %
  % 2500/s 773K
  %
  load FlowSt2500773KLY.dat
  epsEx = FlowSt2500773KLY(:,1)*1.0e-2;
  seqEx = FlowSt2500773KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp2500773 = plot(epsEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp2500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 2500/s 973K
  %
  load FlowSt2500973KLY.dat
  epsEx = FlowSt2500973KLY(:,1)*1.0e-2;
  seqEx = FlowSt2500973KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp2500973 = plot(epsEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp2500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 2500/s 1173K
  %
  load FlowSt25001173KLY.dat
  epsEx = FlowSt25001173KLY(:,1)*1.0e-2;
  seqEx = FlowSt25001173KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp25001173 = plot(epsEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp25001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 2500/s 1373K
  %
  load FlowSt25001373KLY.dat
  epsEx = FlowSt25001373KLY(:,1)*1.0e-2;
  seqEx = FlowSt25001373KLY(:,2);
  %seqEx = seqEx.*(1.0 + epsEx);
  %epsEx = log(1.0 + epsEx);
  pexp25001373 = plot(epsEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp25001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  %[s1, e1] = isoJC(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);
  [s2, e2] = adiJC(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  eps = 0.05;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.1;
  [p1] = intersectPoly(seqEx, epsEx, eps);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));
  eps = 0.2;
  [p2] = intersectPoly(seqEx, epsEx, eps);
  plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p2(1), p2(2));
  
  set(gca, 'XLim', [0 0.35], 'YLim', [0 1000] );
  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
  xlabel('True Strain', 'FontName', 'bookman', 'FontSize', 16);
  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
  legend([pexp2500773 pexp2500973 pexp25001173 pexp25001373], ...
         '2500/s 773 K LY(1997)', ...
         '2500/s 973 K LY(1997)', ...
         '2500/s 1173 K LY(1997)', ...
         '2500/s 1373 K LY(1997)');
  axis square;

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

%
% Isothermal JC data for stress vs strain 
%
function [sig, eps] = isoJC(epdot, T, delT, rho, epmax, Rc)

  E = 213.0e9;
  tmax = epmax/epdot;
  m = tmax/delT;
  ep = 0.0;
  for i=1:m
    sig(i) = JC(epdot, ep, T, Rc);
    eps(i) = ep + sig(i)/E ;
    ep = ep + epdot*delT;
  end

%
% Adiabatic JC data for stress vs strain 
%
function [sig, eps] = adiJC(epdot, T0, delT, rho, epmax, Rc)

  E = 213.0e9;
  tmax = epmax/epdot;
  m = tmax/delT;
  T = T0;
  ep = 0.0;
  for i=1:m
    sig(i) = JC(epdot, ep, T, Rc);
    eps(i) = ep + sig(i)/E;
    ep = ep + epdot*delT;
    Cp = calcCp(T);
    fac = 0.9/(rho*Cp);
    T = T + sig(i)*epdot*fac*delT; 
  end

%
%  Get JC yield stress
%
function [sigy] = JC(epdot, ep, T, Rc)

  %
  % From Johnson and Cook
  %
  %A = 792.0e6;
  %B = 0.6439;
  %C = 0.014;
  %n = 0.26;
  %m = 1.03;
  %Tr = 298.0;
  %Tm = 1793.0;
  %Tc = 1040.0;
  %ep0 = 1.0;

  %
  %  From Lee and Yeh
  %
  %A = 950e6;
  %B = 0.7632;
  %Tr = 298.0;
  %Tm = 1793.0;
  %m = 0.7;
  %if (T > 1040)
  %  m = 0.5;
  %end
  %ep0 = 1.0;
  %epdot = epdot/ep0;
 
  %
  % Compute A from A(Rc) relation
  %
  A1 = 0.0355;
  A2 = 5.5312;
  A = exp(A1*(Rc-2) + A2)*1.0e6;
  B = 0.6439;
  n = 0.26;
  %B = 0.70;
  %n = 0.20;
  C = 0.014;
  %m = 0.7;
  m = 1.03;
  Tr = 298.0;
  %Tr = 100.0;
  Tm = 1793.0;
  if (T < 298)
    m = 1.0;
  elseif (T > 1040)
    m = 0.5;
  end
  ep0 = 1.0;
  epdot = epdot/ep0;

  eppart = A*(1 + B*ep^n);
  if (epdot < 1.0)
    epdotpart = (1 + epdot)^C; 
  else
    epdotpart = 1 + C*log(epdot);
  end
  Tstar = (T - Tr)/(Tm - Tr);
  Tpart = 1 - Tstar^m;

  sigy = eppart*epdotpart*Tpart;

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
  Rc = [32 38 45 49 55]
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

