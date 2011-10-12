
clear;

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

data_new = [1.45938e+06 1.25629e+06 -203610];
data_trial = [1.82286e+06 40787.3 -1.78259e+06];

plot(data_new(:,3)/3.0, -sqrt(3)*data_new(:,2), 'm+', 'MarkerSize', 10, 'LineWidth', 3);
plot(data_trial(:,3)/3.0, -sqrt(3)*data_trial(:,2), 'g+', 'MarkerSize', 8, 'LineWidth', 3);

%trial_I1 = [-1.8833e+06  -1.83868e+06 -1.83824e+06 -1.83823e+06 -1.83823e+06];
%trial_J2 = [669613 670472 670481 670481 670481];
%plot(trial_I1/3.0, -sqrt(3)*trial_J2, 'm+', 'MarkerSize', 8, 'LineWidth', 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           2. crush curve                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2); ylabel('Porosity'); xlabel('I1'); hold on; grid on;
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
