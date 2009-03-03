%_____________________________________________________________
% Function: advectSlabs
% Compute the increment to q in Eq.(3.1.4) of the reference
function q_advected = advectSlabs(xvel_FC, q_slab, ofs, G)
clear j;
global P
if (P.debugSteps)
  fprintf('advectSlabs()\n');
end
q_advected = zeros(G.ghost_Left, G.ghost_Right);
cellVol = G.delX * 1 * 1;                     % delY = delZ = 1
minPrintRange = min(P.printRange);
maxPrintRange = max(P.printRange);

for j = G.first_CC:G.last_CC
  influxVol  = 0.0;
  outfluxVol = 0.0;
  q_outflux  = 0.0;
  q_influx   = 0.0;
  faceFlux_L = 0.0;
  faceFlux_R = 0.0;

  % Logical left face
  if (xvel_FC(j) > 0)
    influxVol  = abs(ofs(j));
    q_influx   = q_slab(j-1);
    faceFlux_L = influxVol * q_influx;
  end
  if (xvel_FC(j) < 0)
    outfluxVol = abs(ofs(j));
    q_outflux  = q_slab(j);
    faceFlux_L = -outfluxVol * q_outflux;
  end
  
  % Logical right face
  if (xvel_FC(j+1) > 0)
    outfluxVol = abs(ofs(j+1));
    q_outflux  = q_slab(j);
    faceFlux_R = -outfluxVol * q_outflux;
  end
  if (xvel_FC(j+1) < 0)
    influxVol  = abs(ofs(j+1));
    q_influx   = q_slab(j+1);
    faceFlux_R = influxVol * q_influx;
  end

  q_advected(j) = (faceFlux_L + faceFlux_R)/cellVol;

  % Print advected quantities near shock front at first timestep
%  if ((j >= minPrintRange) & (j <= maxPrintRange))
%    fprintf('cell %d xvel_FC %16.15E q_influx %16.15E influxVol %16.15E q_outflux %16.15E outfluxVol %16.15E q_advected %15.16E\n', j, xvel_FC(j), q_influx, influxVol, q_outflux, outfluxVol,q_advected(j));
%  end
end
