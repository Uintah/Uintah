function [ofs, rx] = outFluxVol(xvel_FC, delT, delX, nCells)
%_____________________________________________________________
%  Function:  outFluxVol -- computes the slab volume and the
%  corresponding centroid (rx) on the outflux face
%  This assumes that dy and dz = 1.0;
global P
if (P.debugSteps)
    fprintf('outFluxVol()\n');
end
ofs     = zeros(1,nCells);
rx      = zeros(1,nCells);

for j = 1:nCells
    % Slab size lengths and centroid average on the slab
    if xvel_FC(j) >= 0
        delX_slab = xvel_FC(j) * delT;
        rx(j) = delX/2.0 - delX_slab/2.0;
    end

    if xvel_FC(j) < 0
        delX_slab = xvel_FC(j) * delT;
        rx(j) = delX_slab/2.0 - delX/2.0;
    end

    % Outflux volumes
    delY_Z = 1.0;
    ofs(j) = delX_slab * delY_Z;
end
