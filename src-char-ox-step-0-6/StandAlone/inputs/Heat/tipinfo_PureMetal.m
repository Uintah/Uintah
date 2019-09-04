#! /usr/bin/octave -qf

clear all;
close all;
format short e;

global udaFile utilPath makePlot outputFile;

function res = check_input(argv)
  res = length(argv)~=0;
end

function res = check_utilities()
  global utilPath
  res = ~(unix([utilPath 'puda &> /dev/null'])>1 || unix([utilPath 'lineextract &> /dev/null'])>1);
end

function [] = disp_usage()
  printf('tipinfo_PureMetal.m <options>\n')                                                                    
  printf('options:\n')                                                                                       
  printf('  -uda  <udaFileName>  - name of the uda file \n') 
  printf('  -path <utilitesPath> - specify path to Uintah utilites \n')                                                        
  printf('  -plot <true, false>  - produce a plot \n')                                                        
  printf('  -o    <fname>        - Dump the output to a file\n')                                    
end 

function [] = disp_utilities_error()
  disp('Cannot execute the Uintah utilites puda, lineextract');
  disp('  a) make sure you are in the right directory or the specified <utilitesPath> is correct, and');
  disp('  b) the utilities (puda/lineextract) have been compiled');
end

function [] = parse_input(argv)
  global udaFile utilPath makePlot outputFile
  udaFile    = '*.uda';
  utilPath   = './';
  makePlot   = true;
  outputFile = 'tipinfo';

  arg_list = argv ();
  for i = 1:2:nargin
    option    = sprintf("%s",arg_list{i} );
    opt_value = sprintf("%s",arg_list{++i});

    if ( strcmp(option,"-uda") )   
      udaFile = opt_value;
    elseif (strcmp(option,"-path") )
      utilPath = opt_value;
    elseif (strcmp(option,"-plot") )
      makePlot = opt_value;
    elseif (strcmp(option,"-o") )  
      outputFile = opt_value;    
    end                                      
  end

  if utilPath(end) != '/' 
    utilPath = [utilPath '/'];
  end
end

function [TS, T] = get_timesteps()
  global udaFile utilPath
  c0 = sprintf('%spuda -timesteps %s 2> /dev/null | grep "^[0-9]\\+:" > tmp', utilPath, udaFile); s = unix(c0);
  [s,r0] = unix('cut -f 1 -d":" tmp');
  [s,r1] = unix('cut -f 2 -d":" tmp');
  s = unix('rm tmp');
  TS = str2num(r0);
  T = str2num(r1);
end

function [DX, N] = get_gridstats()
  global udaFile utilPath
  c0 = sprintf('%spuda -gridstats %s > tmp 2> /dev/null', utilPath, udaFile); s = unix(c0);
  [s,r0] = unix('grep -m1 dx: tmp | cut -f 2 -d":"');
  [s,r1] = unix('grep -m1 -w "Total Number of Cells" tmp | tr -d [:alpha:] | cut -f 2 -d":"');
  s = unix('rm tmp');
  DX = str2num(r0);
  N = str2num(r1);
end

function [x, y, z] = line_extract(i0, i1, j, k)
  global udaFile utilPath
  c0 = sprintf('%slineextract -v phi -timestep %d -istart %d %d 0 -iend %d %d 0 -nodeCoords -uda %s 2>/dev/null | grep "^-\\?[0-9]\\+"', utilPath, k-1, i0, j, i1, j, udaFile);
  [s, r] = unix(c0);
  data = str2num(r);
  x = data(:,1);
  y = data(:,2);
  z = data(:,4);
end

function [x, y, z] = vert_line_extract(j0, j1, i, k)
  global udaFile utilPath
  c0 = sprintf('%slineextract -v phi -timestep %d -istart %d %d 0 -iend %d %d 0 -nodeCoords -uda %s 2>/dev/null | grep "^-\\?[0-9]\\+"', utilPath, k-1, i, j0, i, j1, udaFile);
  [s, r] = unix(c0);
  data = str2num(r);
  x = data(:,1);
  y = data(:,2);
  z = data(:,4);
end

function [X, Y, Z] = get_interface_neighbour(j, k)
  global iRange
  [x0, y0, z0] = line_extract(iRange(1), iRange(2), j, k);
  intf = find(z0(2:end).*z0(1:end-1)<=0,1);
  if (-z0(intf+1)<z0(intf))
    iRange(1) += intf-2;
    i0 = intf+(-1:3);
    x0 = x0(i0);
    y0 = y0(i0);
    z0 = z0(i0);
  else
    iRange(1) += intf-3;
    i0 = intf+(-2:2);
    x0 = x0(i0);
    y0 = y0(i0);
    z0 = z0(i0);
  end
  [x1, y1, z1] = line_extract(iRange(1), iRange(1)+4, j+1, k);
  [x2, y2, z2] = line_extract(iRange(1), iRange(1)+4, j+2, k);
  X = [x0 x1 x2];
  Y = [y0 y1 y2];
  Z = [z0 z1 z2];
end

function [x, y] = compute_interface_vert_position(X, Y, Z)
  P = polyfit(Y, Z, 4);
  rts = roots(P);
  rts = rts(imag(rts)==0);
  x = X(1);
  y = rts(find(Y(1)<=rts & rts<=Y(5),1));
end

function [X, Y, Z] = get_interface_vert_neighbour(i, k)
  global jRange
  [x0, y0, z0] = vert_line_extract(jRange(1), jRange(2), i, k);
  intf = find(z0(2:end).*z0(1:end-1)<=0,1);
  j0 = intf-3;
  if (-z0(intf+1)<z0(intf))
    j0 += 1;
  end
  if (j0<0)
    j0 = 0;
  end
  jj = j0 + (1:5);
  jRange(1) += intf-2;
  X = x0(jj);
  Y = y0(jj);
  Z = z0(jj);
end

function [x, y] = get_arm(k, N, DX, p)
% figure(1);
  global jRange
  i = N(1)/2 + floor(p/DX(1));
  jRange = [N(2)/2 N(2)];
  x = []; y = [];
  xi = p; yi = 0;
  do
    x = [x, xi];
    y = [y, yi];
    [X, Y, Z] = get_interface_vert_neighbour(i, k);    
%   plot (X,Y,'ko')
    [xi, yi] = compute_interface_vert_position(X, Y, Z);
    i -= 1;
  until ( yi<=y(end) ) 
end

function [x, y] = rm_tip(x, y)
  ind = [(y(2:end)-y(1:end-1))./(x(1:end-1)-x(2:end)) 0] <= 1;
  x = x(ind);
  y = y(ind);
end

function [p, px] = compute_interface_position(X, Z)
% figure(2); 
% plot3(X,0*X,Z, 'ko'); hold on
% view(60,60)
  P = polyfit(X(:,1), Z(:,1), 4);
% xx = linspace(X(1,1),X(5,1),100);
% plot3(xx,0*xx,polyval(P,xx),'m-')  
  rts = roots(P);
  rts = rts(imag(rts)==0);
  p = rts(find(X(2,1)<=rts & rts<=X(4,1),1));
% plot3(p,0,0,'k*')
  px = polyval(polyder(P),p);
% plot3(xx,0*xx,px*(xx-p),'g-')
end

function kloc = compute_interface_local_curvature(X, Y, Z, p, px)
% figure(1);
% plot(p,0,'k*')
  P0 = zeros(5,3);
% figure(2);
% mesh(X, Y, Z);
% mesh(X, -Y, Z);
% axis square
% zlim([-1,1])
% yy = linspace(-Y(1,end),Y(1,end),100);
  for i=1:5  
    P0(i,:) = polyfit(Y(i,:).^2,Z(i,:),2);
%   plot3(X(i,1)+0*yy,yy,polyval(P0(i,:),yy.^2),'c-')
  end;
% figure(3)
% plot(X(:,1),2*P0(:,2),'ko'); hold on;
% xx = linspace(X(1,1),X(5,1),100);
  P1 = polyfit(X(:,1),2*P0(:,2),4);
% plot(xx,polyval(P1,xx),'b-');  
  pyy = polyval(P1,p);
% plot(p,pyy,'m*')
  kloc = -pyy/px^2;
% figure(2)
% plot3(p+0*yy,yy,pyy*yy.^2,'m-')
% plot3(p-kloc/2*yy.^2,yy,0*yy,'g-') 
% hidden('off')
% yy = linspace(0,5,100);
% xlim([0,2*p])
% figure(1);
% yy = linspace(0,5,100);
% plot (p-kloc/2*yy.^2,yy,'g:') 
end

function kpar = compute_interface_parabolic_curvature(k, N, DX, p)
% figure(1);
  [x, y] = get_arm(k, N, DX, p);
% plot(x, y, 'r-');
  [x, y] = rm_tip(x, y);
  P = polyfit(y.^2, x, 1);
  kpar = 2*P(1);
% plot(x, y, 'b*');
% yy = linspace(0,5,100);
% plot(polyval(P,yy.^2),yy,'k:')
% xlim([0 2*P(2)])
% axis equal
% pause
end;

function [P, V, Kloc, Kpar] = compute_interface_info(T, N, DX)
  global udaFile utilPath iRange;

  P = zeros(length(T),1);
  V = zeros(length(T),1);
  Kloc = zeros(length(T),1);
  Kpar = zeros(length(T),1);

  iRange = [N(1)/2 N(1)];
  j = N(2)/2;
 
  for k=1:length(T)
    progress = round(100*k/length(T))
%   figure(1); clf; hold on;
    [X, Y, Z] = get_interface_neighbour(j, k);
%   plot (X,Y,'ro')
    [p, px] = compute_interface_position(X, Z);
%   plot(p,0,'b*')
    v = NaN;
    if(k>1)
      v = (p-P(k-1))/(T(k)-T(k-1));
    end
    kloc = compute_interface_local_curvature(X, Y, Z, p, px);
    kpar = compute_interface_parabolic_curvature(k, N, DX, p);
    P(k) = p;
    V(k) = v;
    Kloc(k) = kloc;
    Kpar(k) = kpar;
  end
end

function [] = write_interface_info(T, P, V, Kloc, Kpar)
  global outputFile
  if (length(outputFile) > 0)
    fid = fopen(outputFile, 'w');
    for k = 1:length(T)
      fprintf(fid,'%g %g %g %g %g \n', T(k), P(k), V(k), Kloc(k), Kpar(k));
    end
    fprintf(fid,'\n');
    fclose(fid);
  end
end

function [] = plot_interface_info(T, P, V, Kloc, Kpar);
  global outputFile
  figure
  plot(T, P);
  xlabel('t')
  ylabel('x')
  title('interface position');
  grid on;
  fname = sprintf([outputFile '_p.eps']);
  print ( fname, '-deps');

  figure
  plot(T, V);
  xlabel('t')
  ylabel('v')
  title('interface velocity');
  grid on;

  fname = sprintf([outputFile  '_v.eps']);
  print ( fname, '-deps');

  figure
  plot(T, Kloc);
  xlabel('t')
  ylabel('{{\kappa}_{loc}}')
  title('interface local curvature');
  grid on;

  fname = sprintf([outputFile  '_kloc.eps']);
  print ( fname, '-deps');

  figure
  plot(T, Kpar);
  xlabel('t')
  ylabel('{{\kappa}_{par}}')
  title('interface parabolic curvature');
  grid on;

  fname = sprintf([outputFile  '_kpar.eps']);
  print ( fname, '-deps');
end
  
%________________________________            

if (~check_input(argv))
  disp_usage();
  exit(-1);
end

parse_input(argv);

if (~check_utilities())
  disp_utilities_error();
  exit(-1);
end

[TS, T] = get_timesteps();

[DX, N] = get_gridstats();

[P, V, Kloc, Kpar] = compute_interface_info(T, N, DX);

write_interface_info(T, P, V, Kloc, Kpar);

if(makePlot)
  plot_interface_info(T, P, V, Kloc, Kpar);
end
  
