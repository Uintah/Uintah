% function: plot_energy_spectrum_cbc
% author:   Tony Saad
% date:     September, 2012
%
% energy_spectrum_cbc: Plots the energy spectrum data reported by
% Comte-Bellot & Corrsin, a.k.a cbc data. You can find the original paper
% here ( http://dx.doi.org/10.1017/S0022112071001599) :
% Comte-Bellot, G. & Corrsin, S. 1971 Simple Eulerian time correlation of 
%     full- and narrow-band velocity signals in grid-generated, ?isotropic? 
%     turbulence. J. Fluid Mech. 48, 273?337.
%
% The data in cbc_spectrum.txt is from table 3 on page 298 of that paper.
%
        
function plot_energy_spectrum_cbc()
  exp = load('cbc_spectrum.txt');
  loglog(exp(:,1)*100,exp(:,2)*1e-6,'k-'); hold on;
  loglog(exp(:,1)*100,exp(:,3)*1e-6,'k-');
  loglog(exp(:,1)*100,exp(:,4)*1e-6,'k-');
  axis([10 1e3 1e-6 1e-3])
end