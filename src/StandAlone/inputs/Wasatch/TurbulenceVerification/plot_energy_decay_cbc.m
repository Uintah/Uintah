% Adapted from FDS' energy_decay.m
% original may be found here:
% https://code.google.com/p/fds-smv/source/browse/trunk/FDS/trunk/Utilities/Matlab/scripts/energy_decay.m.m
%
% function: plot_energy_decay_cbc
% author:   Tony Saad
% date:     June, 2013
%
% plot_energy_decay_cbc: Plots the energy decay data reported by
% Comte-Bellot & Corrsin, a.k.a cbc data. You can find the original paper
% here ( http://dx.doi.org/10.1017/S0022112071001599) :
% Comte-Bellot, G. & Corrsin, S. 1971 Simple Eulerian time correlation of
%     full- and narrow-band velocity signals in grid-generated, ?isotropic?
%     turbulence. J. Fluid Mech. 48, 273?337.
%
% The data in cbc_spectrum.txt is from table 3 on page 298 of that paper.
%

function plot_energy_decay_cbc(N)
L = 0.56549; % \approx 9*(2*pi)/100;

% Gather the Comte-Bellot/Corrsin data
M = load('cbc_spectrum.txt');
k  = M(:,1)*1e2;
E1 = M(:,2)/1e6;
E2 = M(:,3)/1e6;
E3 = M(:,4)/1e6;

% Apply transfer function
k0 = 2*pi/L;
kc = 0.5*N*k0;
delta = pi/kc;
%G = sin(1/2*k.*delta)./(1/2*k.*delta);  % box filter
G = ones(length(k),1);
for j = 1:length(k)
    if k(j)>kc
        tmp = 3*kc/k(j) - 2;
        if tmp > 0
            G(j) = sqrt(tmp); % RJM filter - this one is more accurate
            %G(j) = 0; % spectral cutoff
        else
            G(j) = 0;
        end
    end
end

E1 = G.*G.*E1;
E2 = G.*G.*E2;
E3 = G.*G.*E3;

% Now integrate
E1_bar = trapz(k,E1);
E2_bar = trapz(k,E2);
E3_bar = trapz(k,E3);

hold on;
plot(0.0,  E1_bar,'ko','MarkerSize',7);
plot(0.28, E2_bar,'ro','MarkerSize',7);
plot(0.66, E3_bar,'bo','MarkerSize',7);

% print(gcf,'-dpdf','decay.pdf');
end