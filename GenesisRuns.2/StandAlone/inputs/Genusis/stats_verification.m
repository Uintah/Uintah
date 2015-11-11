% Simple code to calculate turbulence stats for two sine waves

clc
clear all
close all

time = 0:60;
U=sin(time * pi / 180);
V=sin(time * pi / 180 + pi / 3);

data = load('out-f');
U = data(:,2)

var_U = mean(U .^ 2) - mean(U) .^ 2 % Variance
skew_U = mean(U .^ 3) - mean(U) .^ 3 - 3 .* var_U .* mean(U) % Skewness
kurt_U = mean(U .^ 4) - mean(U) .^ 4 - 6 .* var_U .* mean(U) .^ 2 - 4 .* skew_U .* mean(U) % Kurtosis 
%corr_UV = mean(U .* V) - mean(U) .* mean(V) % UV correlation