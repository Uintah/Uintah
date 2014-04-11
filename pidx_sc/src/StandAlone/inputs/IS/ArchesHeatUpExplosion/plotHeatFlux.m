
%_________________________________
% 05/01/07   --Todd
% This matLab script plots the head flux curves
% The format of the "curveFit_coefs_#.dat" files 
% 
% --start file curveFit_coefs_#.dat---
% relative axial location (from low end of the cylinder)
% A
% B
% C
% D
% E
%  ---end file curveFit_coefs_#.dat---
% 
% The coefficients are for a function of the form:
% 
% HT Flux = A + B*sin(theta) + C*cos(theta) + D*sin(2*theta) + E*cos (2*theta)
% Note the bottom of the container is at 270 degrees

close all;
clear all;

%________________________________
% USER INPUTS
%'
dir = 'case11'
dat = {'curveFit_coefs_47.dat','curveFit_coefs_48.dat','curveFit_coefs_49.dat','curveFit_coefs_50.dat','curveFit_coefs_51.dat','curveFit_coefs_52.dat'}
%dat = {'curveFit_coefs_97.dat','curveFit_coefs_98.dat','curveFit_coefs_99.dat','curveFit_coefs_100.dat','curveFit_coefs_101.dat','curveFit_coefs_102.dat'}

legendText = {'47','48', '49', '50','51','52'}
%legendText = {'97', '98', '99','100','101','102'}

symbol = {'-b','-r','-g','-c','-m','-k','.b','.r'};

maxFiles = length(dat)
set(0,'DefaultFigurePosition',[0,0,700,510]);
for( i = 1:maxFiles )
  disp('working on');
  i
  
  f = sprintf('%s/%s',dir,dat{i});
  
  coeff  = importdata(f)
  
  %coeff = [ 0, 45000, 0, 0, 0, 0]
  
  theta= 0:.01:2*pi;
  HT_Flux = coeff(2) + coeff(3)*sin(theta) + coeff(4)*cos(theta) + coeff(5)*sin(2*theta) + coeff(6)*cos(2*theta);
  polar(theta,HT_Flux,symbol{i})
  xlabel('theta')
  hold on
end
legend(legendText);
title (dir);
