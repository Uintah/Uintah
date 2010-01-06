function [ofs, rx] = outFluxVol(xvel_CC, xvel_FC, delT, G)
%_____________________________________________________________
%  Function:  outFluxVol -- computes the slab volume and the
%  corresponding centroid (rx) on the outflux face
%  This assumes that dy and dz = 1.0;
global P
if (P.debugSteps)
  fprintf('outFluxVol()\n');
end
ofs     = zeros(G.ghost_Left,G.ghost_Right);
rx      = zeros(G.ghost_Left,G.ghost_Right);
dx      = G.delX;

for j = G.first_FC:G.last_FC
  % Slab size lengths and centroid average on the slab
  delX_slab = abs(xvel_FC(j) * delT);
  rx(j) = 0.0;
  
  if xvel_FC(j) > 0
    rx(j-1) = dx/2.0 - delX_slab/2.0;
  end

  if xvel_FC(j) < 0
    rx(j) = delX_slab/2.0 - dx/2.0;
  end

  % Outflux volumes
  delY_Z = 1.0;
  ofs(j) = delX_slab * delY_Z;

end
