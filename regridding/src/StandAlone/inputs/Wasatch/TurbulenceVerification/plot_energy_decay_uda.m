% Adapted from FDS' energy_decay.m
% original may be found here:
% https://code.google.com/p/fds-smv/source/browse/trunk/FDS/trunk/Utilities/Matlab/scripts/energy_decay.m.m
%
% function: plot_energy_decay_cbc
% author:   Tony Saad
% date:     June, 2013
%
% plot_energy_decay_uda: Plots the energy decay data reported by Uintah
% data are in the format:
% t   KE
% where t is the simulation time and KE is the total, unaveraged, kinetic
% energy in the domain.
%
        
function plot_energy_decay_uda(filename, N, symbol)
  % Gather the Comte-Bellot/Corrsin data
  M = load(filename);
  t  = M(:,1);
  KE = M(:,2)/(N*N*N); % average ke
  hold on;
  plot(t,  KE, symbol);  
end