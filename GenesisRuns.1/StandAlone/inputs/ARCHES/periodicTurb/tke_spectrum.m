% Adapted from FDS' plotspec_uvw.m
% original may be found here:
% http://code.google.com/p/fds-smv/source/browse/trunk/FDS/trunk/Utilities/
% Matlab/scripts/plotspec_uvw.m
%
% function: tke_spectrum
% author: Tony Saad
% date: Sept 2012
%
% tke_spectrum: calculates the turbulent kinetic energy spectrum given
% u, v, and w velocities. The user must also provide the domain length L.
%
% wave_numbers: an array of wave numbers to be used for plotting
%
% tke_spectrum: the turbulent kinetic energy that correspond to the
% wave_numbers array
%

function [ wave_numbers, tke_spectrum ] = tke_spectrum( u, v, w, L )
  n = size(u,1); % grab the size in each direction - assumes nx = ny = nz
  nt= n*n*n;     % total number of points in spatial domain

  % perform multidimensional fourier transform
  u_hat = fftn(u)/nt;
  v_hat = fftn(v)/nt;
  w_hat = fftn(w)/nt;

  tke_hat=zeros(n,n,n); % allocate memory for spectral tke
  % calculate tubulent kinetic energy
  for k=1:n
    for j=1:n
      for i=1:n
        % NOTE: use conjugate in spectral space
        tke_hat(i,j,k) = 0.5*( u_hat(i,j,k)*conj(u_hat(i,j,k)) + ...
                         v_hat(i,j,k)*conj(v_hat(i,j,k)) + ...
                         w_hat(i,j,k)*conj(w_hat(i,j,k)) );
      end
    end
  end

  % -----------------------------------------------------------------------%
  % spectrum calculation
  % -----------------------------------------------------------------------%
  % For a wave form cos(2*Pi*m*x/L) = cos(k_m x), k0 = 2*Pi/L is the
  % largest wave form (one period) that can be fit into a grid
  k0 = 2*pi/L;
  kmax = n/2; % This is the maximum number of "waves" or peaks that can fit on a grid with npoints
  wave_numbers = k0*[0:n]; % wavenumber array
  tke_spectrum = zeros(size(wave_numbers));

  for kx=1:n
    rkx = kx-1;
    if (kx>kmax+1); rkx=rkx-n; end % conjugate symmetry

    for ky=1:n
      rky = ky-1;
      if (ky>kmax+1); rky=rky-n; end % conjugate symmetry

      for kz=1:n
        rkz = kz-1;
        if (kz>kmax+1); rkz=rkz-n; end % conjugate symmetry
        
        rk = sqrt(rkx^2+rky^2+rkz^2);
        k = round(rk);

        tke_spectrum(k+1) = tke_spectrum(k+1) + tke_hat(kx,ky,kz)/k0;
        
      end
    end
  end
    
end % end function

