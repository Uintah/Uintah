function testMTSSt

  %
  % Compare experimental stress-strain data with MTS model
  %
  plotStressStrain

%
% Compare experimental stress-strain data with MTS model
%
function plotStressStrain

  fid = fopen('StDataRc30Ep0.dat', 'w');
  plotJCRc30(fid);
  fclose(fid);

  fid = fopen('StDataRc32Ep0.dat', 'w');
  plotASMHRc32(fid);
  fclose(fid);

  fid = fopen('StDataRc38Ep0.dat', 'w');
  plotLarsonRc38(fid);
  plotLYRc38500(fid);
  plotLYRc381500(fid);
  plotLYRc382500(fid);
  fclose(fid);

  fid = fopen('StDataRc45Ep0.dat', 'w');
  plotChiRc45_0001(fid);
  plotChiRc45_1000(fid);
  fclose(fid);

  fid = fopen('StDataRc49Ep0.dat', 'w');
  plotChiRc49_0001(fid);
  plotChiRc49_1000(fid);
  fclose(fid);


function plotJCRc30(fid)

  E = 213.0e9;

  fig30 = figure;
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
  pexp0002298 = plot(epEx, seqEx, '^-', 'LineWidth', 2); hold on;
  set(pexp0002298,'LineWidth',2,'MarkerSize',6,'Color',[0.2 0.5 1.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 30.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.5 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.009/s 298K
  %
  load FlowSt0009298KJCShear.dat;
  St298K0009 = FlowSt0009298KJCShear;
  epsEx = St298K0009(:,1)/sqrt(3.0);
  seqEx = St298K0009(:,2)*sqrt(3.0);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp0009298 = plot(epEx, seqEx, 'v-', 'LineWidth', 3); hold on;
  set(pexp0009298,'LineWidth',3,'MarkerSize',6,'Color',[0.1 0.75 1.0]);

  delT = 1.0;
  epdot = 0.009;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 30.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.1 0.75 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.10/s 298K
  %
  load FlowSt010298KJCShear.dat;
  St298K01 = FlowSt010298KJCShear;
  epsEx = St298K01(:,1)/sqrt(3.0);
  seqEx = St298K01(:,2)*sqrt(3.0);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp01298 = plot(epEx, seqEx, '<-', 'LineWidth', 3); hold on;
  set(pexp01298,'LineWidth',3,'MarkerSize',6,'Color',[0.2 0.8 0.2]);

  delT = 0.1;
  epdot = 0.1;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.2 0.8 0.2]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 1.1/s 298K
  %
  load FlowSt1_1298KJCShear.dat;
  St298K1 = FlowSt1_1298KJCShear;
  epsEx = St298K1(:,1)/sqrt(3.0);
  seqEx = St298K1(:,2)*sqrt(3.0);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp1298 = plot(epEx, seqEx, '>-', 'LineWidth', 3); hold on;
  set(pexp1298,'LineWidth',3,'MarkerSize',6,'Color',[0.8 0.4 0.1]);

  delT = 0.01;
  epdot = 1.1;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.4 0.1]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
 
%  set(gca, 'XLim', [0 1.0], 'YLim', [0 1800] );
%  set(gca, 'LineWidth', 3, 'FontName', 'bookman', 'FontSize', 14);
%  xlabel('Plastic Strain', 'FontName', 'bookman', 'FontSize', 16);
%  ylabel('True Stress (MPa) ', 'FontName', 'bookman', 'FontSize', 16);
%  legend([pexp0002298], '0.002/s 298 K (Tension) JC(1985)');
%  legend([pexp0002298 pexp0009298 pexp01298 pexp1298], ...
%         '0.002/s 298 K (Tension) JC(1985)', ...
%         '0.009/s 298 K (Shear) JC(1985)', ...
%         '0.1/s 298 K (Shear) JC(1995)', ...
%         '1.1/s 298 K (Shear) JC(1995)');
%  axis square;

  %
  % 570/s 298K
  %
  load FlowSt570298KJCTen.dat
  epsEx = FlowSt570298KJCTen(:,1);
  seqEx = FlowSt570298KJCTen(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp570298 = plot(epEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp570298,'LineWidth',2,'MarkerSize',6,'Color',[0.3 0.3 0.6]);

  delT = 1.0e-6;
  epdot = 570.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 30;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.3 0.3 0.6]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 604/s 500K
  %
  load FlowSt604500KJCTen.dat
  epsEx = FlowSt604500KJCTen(:,1);
  seqEx = FlowSt604500KJCTen(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp604500 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp604500,'LineWidth',2,'MarkerSize',6,'Color',[0.6 0.3 0.3]);

  delT = 1.0e-6;
  epdot = 604.0;
  T = 500.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 30;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.6 0.3 0.3]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 650/s 735K
  %
  load FlowSt650735KJCTen.dat
  epsEx = FlowSt650735KJCTen(:,1);
  seqEx = FlowSt650735KJCTen(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp650735 = plot(epEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp650735,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-6;
  epdot = 650.0;
  T = 735.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 30;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
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

function plotASMHRc32(fid)

  E = 213.0e9;
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
  %[s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);
  %[s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
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
  eps1 = 0.002 + sigY1/E;
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
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp0002298 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  Rc = 32.0;
  delT = 1.0;
  epdot = 0.002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 422K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(2,2);
  eps1 = 0.002 + sigY1/E;
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
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp0002422 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002422,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0;
  epdot = 0.002;
  T = 422.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 533K Rc 32
  %
  %E = 213.0e9;
  %sigY1 = Aero(3,2);
  %eps1 = 0.002 + sigY1/E;
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
  %[s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  %pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);
  %[s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  %pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  %set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epsEx, eps);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', eps, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 589K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(4,2);
  eps1 = 0.002 + sigY1/E;
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
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp0002589 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp0002589,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0;
  epdot = 0.002;
  T = 589.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 32.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));

  %
  % 0.002/s 644K Rc 32
  %
  E = 213.0e9;
  sigY1 = Aero(5,2);
  eps1 = 0.002 + sigY1/E;
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
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
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

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));

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

function plotLarsonRc38(fid)

  E = 213.0e9;
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
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp00002258 = plot(epEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp00002258,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0002;
  T = 258.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.0002/s 298 K
  %
  load FlowSt0001298KLarson.dat;
  St298K00002 = FlowSt0001298KLarson;
  epsEx = St298K00002(:,1);
  seqEx = St298K00002(:,2)*6.894657;
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp00002298 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp00002298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 1.0 0.2]);

  delT = 10.0;
  epdot = 0.0002;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.2]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 0.0002/s 373 K
  %
  load FlowSt0001373KLarson.dat;
  St373K00002 = FlowSt0001373KLarson;
  epsEx = St373K00002(:,1);
  seqEx = St373K00002(:,2)*6.894657;
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp00002373 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp00002373,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.1 0.1]);

  delT = 10.0;
  epdot = 0.0002;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 38.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.1 0.1]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
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


function plotLYRc38500(fid)

  E = 213.0e9;
  fig50 = figure;
  %set(fig2, 'Position', [378 479 1147 537]);
  %
  % 500/s 298K
  %
  load FlowSt500298KLY.dat
  epsEx = FlowSt500298KLY(:,1)*1.0e-2;
  seqEx = FlowSt500298KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp500298 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 500/s 573K
  %
  load FlowSt500573KLY.dat
  epsEx = FlowSt500573KLY(:,1)*1.0e-2;
  seqEx = FlowSt500573KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp500573 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 500/s 773K
  %
  load FlowSt500773KLY.dat
  epsEx = FlowSt500773KLY(:,1)*1.0e-2;
  seqEx = FlowSt500773KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp500773 = plot(epEx, seqEx, '^-', 'LineWidth', 2); hold on;
  set(pexp500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 0.5]);

  delT = 1.0e-6;
  epdot = 500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 0.5]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
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


function plotLYRc381500(fid)

  E = 213.0e9;
  fig60 = figure;
  %set(fig3, 'Position', [378 479 1147 537]);
  %
  % 1500/s 298K
  %
  load FlowSt1500298KLY.dat
  epsEx = FlowSt1500298KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500298KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp1500298 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp1500298,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 298.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 573K
  %
  load FlowSt1500573KLY.dat
  epsEx = FlowSt1500573KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500573KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp1500573 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp1500573,'LineWidth',2,'MarkerSize',6,'Color',[0.0 0.9 0.2]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 573.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.9 0.2]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 973K
  %
  load FlowSt1500973KLY.dat
  epsEx = FlowSt1500973KLY(:,1)*1.0e-2;
  seqEx = FlowSt1500973KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp1500973 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp1500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 1173K
  %
  load FlowSt15001173KLY.dat
  epsEx = FlowSt15001173KLY(:,1)*1.0e-2;
  seqEx = FlowSt15001173KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp15001173 = plot(epEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp15001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  %
  % 1500/s 1373K
  %
  load FlowSt15001373KLY.dat
  epsEx = FlowSt15001373KLY(:,1)*1.0e-2;
  seqEx = FlowSt15001373KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp15001373 = plot(epEx, seqEx, 'p-', 'LineWidth', 2); hold on;
  set(pexp15001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 1500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
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
         
function plotLYRc382500(fid)

  E = 213.0e9;
  fig70 = figure;
  %set(fig4, 'Position', [378 479 1147 537]);
  %
  % 2500/s 773K
  %
  load FlowSt2500773KLY.dat
  epsEx = FlowSt2500773KLY(:,1)*1.0e-2;
  seqEx = FlowSt2500773KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp2500773 = plot(epEx, seqEx, 'o-', 'LineWidth', 2); hold on;
  set(pexp2500773,'LineWidth',2,'MarkerSize',6,'Color',[0.75 0.25 1.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 773.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.75 0.25 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 2500/s 973K
  %
  load FlowSt2500973KLY.dat
  epsEx = FlowSt2500973KLY(:,1)*1.0e-2;
  seqEx = FlowSt2500973KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp2500973 = plot(epEx, seqEx, 's-', 'LineWidth', 2); hold on;
  set(pexp2500973,'LineWidth',2,'MarkerSize',6,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 973.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 2500/s 1173K
  %
  load FlowSt25001173KLY.dat
  epsEx = FlowSt25001173KLY(:,1)*1.0e-2;
  seqEx = FlowSt25001173KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp25001173 = plot(epEx, seqEx, 'd-', 'LineWidth', 2); hold on;
  set(pexp25001173,'LineWidth',2,'MarkerSize',6,'Color',[0.8 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1173.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.8 0.3 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  %
  % 2500/s 1373K
  %
  load FlowSt25001373KLY.dat
  epsEx = FlowSt25001373KLY(:,1)*1.0e-2;
  seqEx = FlowSt25001373KLY(:,2);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(1);
  pexp25001373 = plot(epEx, seqEx, 'v-', 'LineWidth', 2); hold on;
  set(pexp25001373,'LineWidth',2,'MarkerSize',6,'Color',[0.5 0.3 0.0]);

  delT = 1.0e-6;
  epdot = 2500.0;
  T = 1373.0;
  rhomax = 7850.0;
  epmax = max(epsEx);
  Rc = 38;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.5 0.3 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(1);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
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
         
function plotChiRc45_0001(fid)

  E = 213.0e9;
  fig01 = figure;

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
  pexp00001173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp00001173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001298KChi.dat
  epsEx = FlowSt0001298KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001298KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp00001298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp00001298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001373KChi.dat
  epsEx = FlowSt0001373KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001373KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp00001373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp00001373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
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


function plotChiRc45_1000(fid)

  E = 213.0e9;
  fig11 = figure;

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
  pexp1000173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp1000173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000298KChi.dat
  epsEx = FlowSt1000298KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000298KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp1000298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp1000298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000373KChi.dat
  epsEx = FlowSt1000373KChi(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000373KChi(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp1000373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp1000373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 45.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
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

function plotChiRc49_0001(fid)

  E = 213.0e9;
  fig00 = figure;

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
  pexp00001173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp00001173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001298KChi2.dat
  epsEx = FlowSt0001298KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001298KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp00001298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp00001298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
  load FlowSt0001373KChi2.dat
  epsEx = FlowSt0001373KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt0001373KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp00001373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp00001373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 10.0;
  epdot = 0.0001;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s1, e1] = isoMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatiso  = plot(e1, s1*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatiso,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.1;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.2;
  %[p2] = intersectPoly(seqEx, epEx, ep);
  %plot(p2(1),p2(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p2(1), p2(2));
  
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

function plotChiRc49_1000(fid)

  E = 213.0e9;
  fig10 = figure;

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
  pexp1000173 = plot(epEx, seqEx, 'r.-', 'LineWidth', 2); hold on;
  set(pexp1000173,'LineWidth',2,'MarkerSize',9,'Color',[1.0 0.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 173.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[1.0 0.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000298KChi2.dat
  epsEx = FlowSt1000298KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000298KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E;
  epEx = epEx - epEx(2);
  pexp1000298 = plot(epEx, seqEx, 'g.-', 'LineWidth', 2); hold on;
  set(pexp1000298,'LineWidth',2,'MarkerSize',9,'Color',[0.0 1.0 0.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 298.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 1.0 0.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
  load FlowSt1000373KChi2.dat
  epsEx = FlowSt1000373KChi2(:,1)/sqrt(3)*1.0e-2;
  seqEx = FlowSt1000373KChi2(:,2)*sqrt(3);
  epEx = epsEx - seqEx*1.0e6/E - 0.003;
  epEx = epEx - epEx(2);
  pexp1000373 = plot(epEx, seqEx, 'b.-', 'LineWidth', 2); hold on;
  set(pexp1000373,'LineWidth',2,'MarkerSize',9,'Color',[0.0 0.0 1.0]);

  delT = 1.0e-6;
  epdot = 1000.0;
  T = 373.0;
  rhomax = 7831.0;
  epmax = max(epsEx);
  Rc = 49.0;
  [s2, e2] = adiMTS(epdot, T, delT, rhomax, epmax, Rc);
  pmatadi  = plot(e2, s2*1.0e-6, 'k-.', 'LineWidth', 2); hold on;
  set(pmatadi,'LineWidth',2,'MarkerSize',6, 'Color',[0.0 0.0 1.0]);

  ep = 0.0;
  p1(1) = 0.0;
  p1(2) = seqEx(2);
  plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  %ep = 0.05;
  %[p1] = intersectPoly(seqEx, epEx, ep);
  %plot(p1(1),p1(2),'kx','MarkerSize', 10, 'LineWidth', 3);
  %fprintf(fid, '%g %g %g %g %g %g\n', ep, epdot, T, Rc, p1(1), p1(2));
  
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
  % Compute sigma_es
  %
  %sigma_es0 = 790e6;
  %g_0es = 0.112;
  %g_0es = 5.85e-3*Rc - 8.92e-3;
  %sigma_es0 = -28.1*Rc + 1688.0;
  g_0es = 5.031e-4*Rc^3 - 5.74e-2*Rc^2 + 2.1451*Rc - 26.054;
  sigma_es0 = 0.70417*Rc^3  - 85.561*Rc^2 + 3377.3*Rc - 42583.0;
  sigma_es0 = sigma_es0*1.0e6;
  edot_0es = 1.0e7;
  kappa = 1.38e-23;
  b = 2.48e-10;
  sigma_es = sigma_es0*(epdot/edot_0es)^(kappa*T/(mu*b^3*g_0es)); 

  %
  % Compute theta_0
  %
  %a_0 = 5102.4e6;
  %a_1 = 0.0;
  %a_2 = 0.0;
  %a_3 = 2.0758e6;
  a_0 = -7.1011e+01*Rc^3 + 8.1373e+03*Rc^2 - 3.0405e+05*Rc + 3.7107e+06;
  a_1 =  8.0079e-02*Rc^3 - 1.0126e+01*Rc^2 + 4.2081e+02*Rc - 5.7290e+03;
  a_2 = -5.0786e-01*Rc^3 + 6.1808e+01*Rc^2 - 2.4713e+03*Rc + 3.2441e+04;
  a_3 =  2.8783e-01*Rc^3 - 3.3058e+01*Rc^2 + 1.2407e+03*Rc - 1.5209e+04;
  theta_0 = a_0 + a_1*log(epdot) + a_2*sqrt(epdot) + a_3*T;
  theta_0 = theta_0*1.0e6;

  aIV_0 =  8.7787e+00*Rc^3 - 1.0376e+03*Rc^2 + 3.9919e+04*Rc - 4.9911e+05;
  aIV_1 =  1.5622e-02*Rc^3 - 2.0127e+00*Rc^2 + 8.5606e+01*Rc - 1.2001e+03;
  aIV_2 =  1.6347e-01*Rc^3 - 1.8436e+01*Rc^2 + 6.7415e+02*Rc - 7.9722e+03;
  aIV_3 = -2.8602e-02*Rc^3 + 3.3571e+00*Rc^2 - 1.2801e+02*Rc + 1.5858e+03;
  theta_IV = aIV_0 + aIV_1*log(epdot) + aIV_2*sqrt(epdot) + aIV_3*T;
  theta_IV = theta_IV*1.0e6;

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
  % Compute s_i
  %
  g_0i = -1.5425e-3*Rc^3 + 2.0396e-1*Rc^2 - 8.8536*Rc + 1.27e2;
  epdot_0i = 1.0e8;
  p_i = 2.0/3.0;
  q_i = 1.0;
  s_i = (1.0 - (kappa*T/(mu*b^3*g_0i)*log(epdot_0i/epdot))^(1/q_i))^(1/p_i);

  %
  % Compute sigma/mu
  %
  sigma_a = 50.0e6;

  %
  % Compute sigma_i
  %
  sigma_i = 0.18162*Rc^3 - 24.029*Rc^2 + 1077.1*Rc - 14721;
  sigma_i = sigma_i*1.0e6;

  sigma_mu = sigma_a/mu + s_i*sigma_i/mu_0 + s_e*sigma_e/mu_0;

  %
  % Compute sigy
  %
  sigy = sigma_mu*mu;

%
% Integrate dsigma_e/dep
%
function [sigma_e] = computeSigma_e(theta_0, theta_IV, sigma_es, ep)

  if (ep == 0)
    sigma_e = 0.0;
    return;
  end

  alpha = 2.0;
  sigma_e = 0;
  dep = ep/100;
  for i=1:101
    FX = tanh(alpha*sigma_e/sigma_es)/tanh(alpha);
    sigma_e = sigma_e + dep*(theta_0*(1-FX) + theta_IV*FX);
    if (sigma_e > sigma_es)
      sigma_e = sigma_es;
      break;
    end
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
% Compute the tangent modulus and sigma_e (isothermal)
%
function [theta, sigma_e, eplas, sig_e] = ...
          computeTangentModulusIso(delT, ep, sig, epdot, T0, Rc)

  [n, m] = size(ep);
  count = 1;
  for i=1:n
    if ~(ep(i) < 0.0)
      sig_m = sig(i);
      sig_e(count) = computeSige(sig_m, epdot, T0, T0, Rc);
      eplas(count) = ep(i);
      count = count+1;
    end
  end
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
function [theta, sigma_e, eplas, sig_e] = ...
          computeTangentModulusAdi(delT, ep, sig, epdot, T0, Rc)

  [n, m] = size(ep);
  count = 1;
  for i=1:n
    if ~(ep(i) < 0.0)
      sig_m = sig(i);
      ep_m =  ep(i);
      T = computeTemp(delT, sig_m, ep_m, epdot, T0);
      sig_e(count) = computeSige(sig_m, epdot, T0, T0, Rc);
      eplas(count) = ep(i);
      count = count+1;
    end
  end
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
function [sigma_e] = computeSige(sig_y, epdot, T, T0, Rc)

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
  S_e = (1.0 - (kappa*T/(mu*b^3*g_0e)*log(edot_0e/epdot))^(1/q_e))^(1/p_e);

  %
  % Compute sig_e
  %
  sigma_e = 1.0/S_e*(mu_0/mu*(sig_y - sig_a) - S_i*sigma_i);
  
  %[sigma_e mu_0 mu sig_y sig_a S_i sigma_i]
