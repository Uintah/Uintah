function plotCamClay
  clear;
  close all;
  plotCamClaySimData

function plotCamClayP
  epse_v = -0.05:0.001: 0.05;
  epse_s = 0.0;
  for ii=1:length(epse_v)
    p(ii) = computeP(epse_v(ii), epse_s);
  end
  plot(epse_v, p);
  

function [p] = computeP(epse_v, epse_s)

  p0 = -9.0e4;
  alpha = 60;
  kappatilde = 0.018;
  epse_v0 = 0.0;
  beta = 1.0 + 1.5*alpha/kappatilde*epse_s*epse_s;
  p = p0*beta*exp(-(epse_v - epse_v0)/kappatilde);

function [q] = computeQ(epse_v, epse_s)

  mu = computeShearModulus(epse_v, epse_s);
  q = 3.0*mu*epse_s;

function [mu] = computeShearModulus(epse_v, epse_s)

  mu0 = 5.4e6;
  p0 = -9.0e4;
  alpha = 60;
  kappatilde = 0.018;
  epse_v0 = 0.0;
  mu = mu0 - alpha*p0*exp(-(epse_v - epse_v0)/kappatilde);
  
function [f] = camclayYield(p, q, pc)

  M = 1.05;
  f = q^2/M^2 + p*(p - pc);
  
function [q] = camclayYieldQ(p, pc)

  M = 1.05;
  q = sqrt(-M^2*p*(p - pc));


  
function plotCamClayYieldFunction(pc, color)

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Draw the analytical data:                 %
  %           1. q-p plot (stress trajectory) %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %           1. q-p plot (stress trajectory) %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  pp_min = 1.0;
  pp_max = pc;
  pp = linspace(pp_min, pp_max, 100);
  for ii=1:length(pp)
    qq(ii) = camclayYieldQ(pp(ii), pc);
  end
  figure(1); hold on;
  plot(pp, qq, '-', 'LineWidth', 2, 'Color', color); 
  plot(pp, -qq, '-', 'LineWidth', 2, 'Color', color); 

function plotCamClaySimData

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Draw numerical results:                                %
  %           1. q-p plot (stress trajectory)              %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % iter p q pc dfdp dfdq fyield
  iterData =  ...
   [[ 0 -81678 38761 -90000 0 0 6.83015e+08];...
    [ 1 -76022.8 4636.76  -89079.3 -62966.2 8411.35 -9.73097e+08];...
    [ 2 -93753.2 31693.8  -92065.3 -95441.2 57494.4 1.06936e+09];...
    [ 3 -1.30854e+06 240747  -140631 -2.47644e+06 436730 1.58082e+12];...
    [ 4 -812477 138519  -130282 -1.49467e+06 251281 5.71671e+11];...
    [ 5 -512063 71904.3  -120994 -903133 130439 2.04942e+11];...
    [ 6 -330158 28031  -112781 -547535 50850 7.24814e+10];...
    [ 7 -219556 2725.99  -105636 -333477 4945.11 2.50188e+10];...
    [ 8 -152827 2138.36  -99660.3 -205993 3879.12 8.12942e+09];...
    [ 9 -115456 1809.27  -95268.6 -135643 3282.12 2.33369e+09];...
    [ 10 -97913.2 1654.79  -92778.4 -103048 3001.89 5.05255e+08];...
    [ 11 -93341.1 1614.53  -92068 -94614.1 2928.85 1.21189e+08];...
    [ 12 -94234.3 1622.39  -92209.1 -96259.5 2943.12 1.93233e+08];...
    [ 13 -94173.2 1621.85  -92199.5 -96147 2942.14 1.8826e+08];...
    [ 14 -94193.1 1622.03  -92202.6 -96183.6 2942.46 1.89876e+08];...
    [ 15 -94187 1621.98  -92201.6 -96172.3 2942.36 1.89381e+08];...
    [ 16 -94188.9 1621.99  -92201.9 -96175.8 2942.39 1.89533e+08];...
    [ 17 -94188.3 1621.99  -92201.8 -96174.7 2942.38 1.89486e+08];...
    [ 18 -94188.5 1621.99  -92201.9 -96175 2942.38 1.89501e+08];...
    [ 19 -94188.4 1621.99  -92201.9 -96174.9 2942.38 1.89496e+08];...
    [ 20 -94188.4 1621.99  -92201.9 -96175 2942.38 1.89497e+08]];

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %           1. q-p plot (stress trajectory)              %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  figure(1); ylabel('q'); xlabel('p'); hold on; grid on;
  pSim = iterData(:,2);
  qSim = iterData(:,3);
  pcSim = iterData(:,4);
  for ii=1:length(pSim)
   qSimYield(ii) = camclayYieldQ(pSim(ii), pcSim(ii));
  end
  plot(pSim,qSim,'-bx','LineWidth',2, 'MarkerSize', 8);
  plot(pSim,qSimYield,'rs','LineWidth',2, 'MarkerSize', 8);
  plot(pSim(1),qSim(1),'go','MarkerSize',10, 'LineWidth',5);
  plot(pSim(length(pSim)),qSim(length(pSim)),'ro','MarkerSize',10, 'LineWidth',5);
  %axis equal

  plotCamClayYieldFunction(pcSim(1), [0 1 0]);
  for ii=2:length(pSim)-1
    plotCamClayYieldFunction(pcSim(ii), [0.75 0.25 0.1]);
  end
  plotCamClayYieldFunction(pcSim(length(pSim)), [1 0 0]);

function geoModel
%-------------------------------------------------------------
% Parameters
%-------------------------------------------------------------
f_slope = 0.057735026919;
I1_peak = 6.123724356953976e2;
%cap_ratio = 0.001;
p0_crush = 1837.0724e3;
p1_crush = 6.6666666666666666e-4;
cap_ratio = 14.8;
%p0_crush = 1837.0724;

kappa0 = (-p0_crush + cap_ratio*f_slope*I1_peak)/(cap_ratio*f_slope + 1.0)
cap_rad = -cap_ratio*f_slope*(kappa0 - I1_peak)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the input data from text files:                           %
%           1. time history of stresse                           %
%           2. time history of volumetric part of plastic strain %
%           3. time history of volumetric part of elastic strain %
%           4. time history of kappa                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           1. time history of stresse                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin1=fopen('outStress.txt','r');
t=[]; stress=[]; PlasStrain=[]; ElasStrain=[]; kappa=[];
word=fscanf(fin1,'%s1');
while word
    t=[t str2num(word)];
    fscanf(fin1,'%s1'); fscanf(fin1,'%s1'); fscanf(fin1,'%s1');
    stress=[stress [str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'))
        str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'))
        str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'));str2num(fscanf(fin1,'%s1'))]];
    word=fscanf(fin1,'%s1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. time history of volumetric part of plastic strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin2=fopen('outPlasticStrain.txt','r');
word=fscanf(fin2,'%s1');
while word
    fscanf(fin2,'%s1'); fscanf(fin2,'%s1'); fscanf(fin2,'%s1');
    PlasStrain=[PlasStrain str2num(fscanf(fin2,'%s1'))];
    word=fscanf(fin2,'%s1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           3. time history of volumetric part of elastic strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin3=fopen('outElasticStrain.txt','r');
word=fscanf(fin3,'%s1');
while word
    fscanf(fin3,'%s1'); fscanf(fin3,'%s1'); fscanf(fin3,'%s1');
    ElasStrain=[ElasStrain str2num(fscanf(fin3,'%s1'))];
    word=fscanf(fin3,'%s1');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           4. time history of kappa                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fin4=fopen('outKappa.txt','r');
word=fscanf(fin4,'%s1');
while word
    fscanf(fin4,'%s1'); fscanf(fin4,'%s1'); fscanf(fin4,'%s1');
    kappa=[kappa str2num(fscanf(fin4,'%s1'))];
    word=fscanf(fin4,'%s1');
end
fclose('all');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw the analytical data:                 %
%           1. q-p plot (stress trajectory) %
%           2. crush curve                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           1. q-p plot (stress trajectory) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pp_min = kappa0/3.0
pp_max = 1.5*I1_peak/3.0
pp = linspace(pp_min, pp_max, 100);
qq = -sqrt(3.0)*f_slope*(3.0*pp - I1_peak);
figure(1); hold on;
plot(pp, qq, '-m', 'LineWidth', 2); 
plot(pp, -qq, '-m', 'LineWidth', 2); 

%plot(pp,61.238-0.3*pp,'-g','LineWidth',3);
%plot(pp,-61.238+0.3*pp,'-g','LineWidth',3);

pp_min = (kappa0 - cap_rad)/3.0
pp_max = kappa0/3.0
p_cap=linspace(pp_min, pp_max, 100);
num = kappa0 - 3.0*p_cap;
den = cap_rad;
ratio = num/den;
beta_cap=sqrt(1-ratio.^2);
q_cap = -sqrt(3.0)*f_slope*(3.0*p_cap - I1_peak).*beta_cap;
plot(p_cap, q_cap, '-m', 'LineWidth', 2); 
plot(p_cap, -q_cap, '-m', 'LineWidth', 2); 

pp_max = (kappa0 - cap_rad)/3.0
pp_min = 1.1*(kappa0 - cap_rad)/3.0
p_out_cap=linspace(pp_min, pp_max, 5);
num = kappa0 - 3.0*p_out_cap;
den = cap_rad;
ratio = num/den;
beta_out_cap=sqrt(ratio.^2-1);
beta_out_cap = 0;
%q_out_cap = -sqrt(3.0)*f_slope*(3.0*p_out_cap - I1_peak).*beta_out_cap;
q_out_cap = -sqrt(3.0)*(kappa0-3.0*p_out_cap - cap_rad)
plot(p_out_cap, q_out_cap, '-r', 'LineWidth', 2); 
plot(p_out_cap, -q_out_cap, '-r', 'LineWidth', 2); 

%pcap=linspace(-612.35,-204.12,100);
%Beta=sqrt(1-((3*pcap+612.35)/1224.8).^2);
%plot(pcap,Beta*61.238-0.3*Beta.*pcap,'-g','LineWidth',3);
%plot(pcap,-Beta*61.238+0.3*Beta.*pcap,'-g','LineWidth',3);

kappa_n = kappa(length(kappa));
cap_rad_n = -cap_ratio*f_slope*(kappa_n - I1_peak)

pp_min = kappa_n/3.0
pp_max = 1.5*I1_peak/3.0
pp = linspace(pp_min, pp_max, 100);
qq = -sqrt(3.0)*f_slope*(3.0*pp - I1_peak);
figure(1); hold on;
plot(pp, qq, '-@m', 'LineWidth', 2); 
plot(pp, -qq, '-@m', 'LineWidth', 2); 

%plot(pp,61.238-0.3*pp,'-g','LineWidth',3);
%plot(pp,-61.238+0.3*pp,'-g','LineWidth',3);

pp_min = (kappa_n - cap_rad_n)/3.0
pp_max = kappa_n/3.0
p_cap=linspace(pp_min, pp_max, 100);
num = kappa_n - 3.0*p_cap;
den = cap_rad_n;
ratio = num/den;
beta_cap=sqrt(1-ratio.^2);
q_cap = -sqrt(3.0)*f_slope*(3.0*p_cap - I1_peak).*beta_cap;
plot(p_cap, q_cap, '-@m', 'LineWidth', 2); 
plot(p_cap, -q_cap, '-@m', 'LineWidth', 2); 

pp_max = (kappa_n - cap_rad_n)/3.0
pp_min = 1.1*(kappa_n - cap_rad_n)/3.0
p_out_cap=linspace(pp_min, pp_max, 5);
num = kappa_n - 3.0*p_out_cap;
den = cap_rad_n;
ratio = num/den;
beta_out_cap=sqrt(ratio.^2-1);
beta_out_cap = 0;
%q_out_cap = -sqrt(3.0)*f_slope*(3.0*p_out_cap - I1_peak).*beta_out_cap;
q_out_cap = -sqrt(3.0)*(kappa_n-3.0*p_out_cap - cap_rad_n)
plot(p_out_cap, q_out_cap, '-r', 'LineWidth', 2, 'LineStyle', '--'); 
plot(p_out_cap, -q_out_cap, '-r', 'LineWidth', 2, 'LineStyle', '--'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. crush curve                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2); 
p00 = linspace(-p0_crush,5*-p0_crush,100);
plot(p00, 0.5*exp(p1_crush*(p00+p0_crush)),'-g','LineWidth',3);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw numerical results:                                %
%           1. q-p plot (stress trajectory)              %
%           2. crush curve                               %
%           3. time history of kappa                     %
%           4. pressure versus volumetric part of strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           1. q-p plot (stress trajectory)              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1); ylabel('q'); xlabel('p'); hold on; grid on;
p=(stress(1,:)+stress(5,:)+stress(9,:))/3;
sDev1=stress(1,:)-p; sDev2=stress(2,:);   sDev3=stress(3,:);
sDev4=stress(4,:);   sDev5=stress(5,:)-p; sDev6=stress(6,:);
sDev7=stress(7,:);   sDev8=stress(8,:);   sDev9=stress(9,:)-p;
J2=1/2*(sDev1.^2+sDev5.^2+sDev9.^2) ...
    +sDev2.^2+sDev6.^2+sDev3.^2;
J3=sDev1.*sDev5.*sDev9+2*sDev2.*sDev6.*sDev3 ...
    -sDev2.^2.*sDev9-sDev6.^2.*sDev5-sDev3.^2.*sDev1;
q=sqrt(3*J2).*sign(J3);
plot(p,q,'-bx','LineWidth',2, 'MarkerSize', 8);
plot(p(1),q(1),'go','MarkerSize',10, 'LineWidth',5);
plot(p(length(p)),q(length(p)),'ro','MarkerSize',10, 'LineWidth',5);

data_new = ...
[[-65656.5 39448.7 -1.81986e+06];...
[-65655.3 39449.8 -1.81986e+06];...
[-65659.1 39446.2 -1.81986e+06];...
[-65855.5 39115.8 -1.81754e+06];...
[-65857.6 39113.8 -1.81755e+06];...
[-65905.6 39032.3 -1.81697e+06];...
[-65906.4 39031.5 -1.81697e+06];...
[-65918 39011.5 -1.81682e+06];...
[-65918.3 39011.2 -1.81682e+06];...
[-65921.1 39006.3 -1.81678e+06];...
[-65923.2 39002.7 -1.81676e+06];...
[-65923.5 39002.1 -1.81675e+06];...
[-65923.5 39002.1 -1.81675e+06];...
[-65923.5 39002.1 -1.81675e+06];...
[-65923.5 39002.1 -1.81675e+06];...
[-65923.5 39002 -1.81675e+06];...
[-65923.5 39002 -1.81675e+06];...
[-65923.5 39002 -1.81675e+06];...
[-65923.5 39002 -1.81675e+06];...
[-65923.5 39002 -1.81675e+06]];

data_trial = ...
[[1917.37 39652.1 -1.82115e+06];...
[1918.61 39653.3 -1.82115e+06];...
[1914.57 39649.3 -1.82115e+06];...
[482.703 39167.7 -1.81787e+06];...
[480.485 39165.5 -1.81787e+06];...
[128.997 39046.2 -1.81705e+06];...
[128.144 39045.3 -1.81705e+06];...
[41.0243 39015.9 -1.81685e+06] ;...
[40.7378 39015.6 -1.81685e+06];...
[19.1215 39008.4 -1.8168e+06];...
[2.91462 39003 -1.81676e+06];...
[0.213918 39002.1 -1.81675e+06];...
[0.202716 39002.1 -1.81675e+06];...
[0.196033 39002.1 -1.81675e+06];...
[0.192167 39002.1 -1.81675e+06];...
[0.0283575 39002 -1.81675e+06];...
[0.0272511 39002 -1.81675e+06];...
[0.0266297 39002 -1.81675e+06];...
[0.00672218 39002 -1.81675e+06];...
[0.00653669 39002 -1.81675e+06]];

plot(data_new(:,3)/3.0, -sqrt(3)*data_new(:,2), 'm+', 'MarkerSize', 10, 'LineWidth', 3);
plot(data_trial(:,3)/3.0, -sqrt(3)*data_trial(:,2), 'g+', 'MarkerSize', 8, 'LineWidth', 3);

%trial_I1 = [-1.8833e+06  -1.83868e+06 -1.83824e+06 -1.83823e+06 -1.83823e+06];
%trial_J2 = [669613 670472 670481 670481 670481];
%plot(trial_I1/3.0, -sqrt(3)*trial_J2, 'm+', 'MarkerSize', 8, 'LineWidth', 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. crush curve                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2); ylabel('Vol. plastic strain+0.5'); xlabel('I1'); hold on; grid on;
plot(3*p,PlasStrain+0.5,'-rx','LineWidth',2, 'MarkerSize',8);
plot(3*p(1),PlasStrain(1)+0.5,'go','MarkerSize',10, 'LineWidth',5);
plot(3*p(length(p)),PlasStrain(length(p))+0.5,'ro','MarkerSize',10, 'LineWidth',5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           3. time history of kappa                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(3); ylabel('Kappa'); xlabel('Time'); hold on; grid on;
plot(t,kappa,'-b','LineWidth',2);
plot(t(1),kappa(1),'go','MarkerSize',10, 'LineWidth',5);
plot(t(length(t)),kappa(length(t)),'ro','MarkerSize',10, 'LineWidth',5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           4. pressure versus volumetric part of strain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(4); ylabel('pressure'); xlabel('VolumetricStrain'); hold on; grid on;
plot(-ElasStrain-PlasStrain,-3*p,'-b','LineWidth',2);
plot(-ElasStrain(1)-PlasStrain(1),-3*p(1),'go','MarkerSize',10, 'LineWidth',5);
plot(-ElasStrain(length(ElasStrain))-PlasStrain(length(ElasStrain)),-3*p(length(ElasStrain)),'ro','MarkerSize',10, 'LineWidth',5);
