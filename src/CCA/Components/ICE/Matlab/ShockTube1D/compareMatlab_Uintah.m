% 1) Run the matlab script ice.m at a fixed timestep size
%   - copy the final timestep
%
% 2) Run the sus shock tube problem and the same timestep/resolution
%   - Make sure you set the final timestep as was computed in 1)
%   - set tstep to the last timestep of the sus simulation
% 


clear all;
close all;
setenv('LD_LIBRARY_PATH', ['/usr/lib']);

tstep = 1
delT = 1;

loadUintah      % load uintah data into arrays

ML_CC     = importdata('matlab_CC_100.dat', ' ', 2);
ML_FC     = importdata('matlab_FC_100.dat', ' ', 2);

x_CC          = ML_CC.data(:,1);
press_eq_CC   = ML_CC.data(:,2);
delPDilatate  = ML_CC.data(:,3);
press_CC      = ML_CC.data(:,4);
xvel_CC       = ML_CC.data(:,5);
temp_CC       = ML_CC.data(:,6);
rho_CC        = ML_CC.data(:,7);

x_FC          = ML_FC.data(:,1);
xvel_FC       = ML_FC.data(:,2);
press_FC      = ML_FC.data(:,3);

%______________________________________________________________________
% make the plots 
figure(1);
set(gcf,'position',[100,600,900,900]);

titleStr= {'Shock Tube','Uintah vs Matlab ICE', 'Advection Scheme: Uintah, 1st order, Matlab 2nd order' '200 Cells'};

xlo = 0.2;
xhi = 0.9;

subplot(2,2,1), plot(x_CC ,rho_CC,'+', x_ice, rho_ice, 'o');
xlim([xlo xhi]);
legend('\rho', '\rho Uintah',2);
title(titleStr);
grid on;

subplot(2,2,2), plot(x_CC ,xvel_CC,'+', x_ice, vel_ice, 'o');
xlim([xlo xhi]);
legend('U', 'U Uintah',2);
grid on;

subplot(2,2,3), plot(x_CC ,temp_CC,'+', x_ice, temp_ice, 'o');
xlim([xlo xhi]);
legend('Temperature', 'Temperature Uintah',2);
grid on;

subplot(2,2,4), plot(x_CC ,press_CC,'+', x_ice, press_ice, 'o');
xlim([xlo xhi]);
legend('Pressure', 'Pressure Uintah',2);
grid on;

figure(2);
set(gcf,'position',[100,100,900,900]);

subplot(3,1,1), plot(x_FC, xvel_FC,'+', x_FC_ice, uvel_FC_ice, 'o');
xlim([xlo xhi]);
title(titleStr);
legend('xvel FC','uvel FC Uintah',2);
grid on;

subplot(3,1,2), plot(x_CC ,delPDilatate, x_ice, delP_ice);
xlim([xlo xhi]);
legend('delP','delP Uintah',2);
grid on;

subplot(3,1,3), plot(x_CC, press_eq_CC,'+', x_ice, press_eq_ice,'o');
xlim([xlo xhi]);
legend('press equilibration','press equilibration Uintah',2);
grid on; 

input('hit return')
figure(1);
print -depsc comparePlots1.eps
figure(2);
print -depsc comparePlots2.eps
