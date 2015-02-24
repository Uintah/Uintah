% function: plot_energy_spectrum_uvw
% author: Tony Saad
% date: Sept 2012
%
% plot_energy_spectrum_uvw: calculates the turbulent kinetic energy spectrum given
% (u,v,w) velocities in a single csv file. 
%

function plot_energy_spectrum_uvw(uvwFileName, L, marker_format)

  M = csvread(uvwFileName);  
  s = size(M);
  n = round(s(1)^(1/3));

  % convert to 3D array
  p=0;
  u = zeros(n,n,n);
  v = zeros(n,n,n);
  w = zeros(n,n,n);
  tke=zeros(n,n,n);

  for k=1:n
    for j=1:n
      for i=1:n
        p=p+1;
        u(i,j,k) = M(p,1);
        v(i,j,k) = M(p,2);
        w(i,j,k) = M(p,3);
        %tke(i,j,k) = 0.5*(u(i,j,k)^2+v(i,j,k)^2+w(i,j,k)^2);            
      end
    end
  end
  
  % calculate turbulent kinetic energy spectrum
  [wn,vt]=tke_spectrum(u,v,w,L); 

  % plot the energy spectrum
  hold on; 
  loglog(wn(2:n),vt(2:n),marker_format,'MarkerSize',15);
  
end