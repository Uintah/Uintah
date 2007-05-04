
%_________________________________
% 05/01/07   --Todd
% This matLab script plots the head flux curves
% The format of the "polynomial_#.dat" files 
% 
% --start file polynomial_#.dat---
% relative axial location (from low end of the cylinder)
% A
% B
% C
% D
% E'polynomial_100.dat','polynomial_101.dat'
%  ---end file polynomial_#.dat---
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
dir = 'case3'
%dat = {'polynomial_46.dat','polynomial_47.dat', 'polynomial_48.dat','polynomial_49.dat','polynomial_50.dat','polynomial_51.dat','polynomial_52.dat','polynomial_53.dat'}
dat = {'polynomial_96.dat','polynomial_97.dat','polynomial_98.dat','polynomial_99.dat','polynomial_100.dat','polynomial_101.dat','polynomial_102.dat','polynomial_103.dat'}

%legendText = {'46','47','48', '49', '50','51','52','53'}
legendText = {'96','97', '98', '99','100','101','102','103'}

symbol = {'-b','-r','-g','-c','-m','-k','.b','.r'};

maxFiles = length(dat)

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