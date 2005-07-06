%_____________________________________________________________
% Function: advectSlabs
% Compute the increment to q in Eq.(3.1.4) of the reference
function q_advected = advectSlabs(xvel_FC, q_slab, ofs, nCells,delX)
clear j;
global P
if (P.debugSteps)
    fprintf('advectSlabs()\n');
end
q_advected = zeros(1,nCells + 2);
cellVol = delX * 1 * 1;                     % delY = delZ = 1
minPrintRange = min(P.printRange);
maxPrintRange = max(P.printRange);

% Note that the following has to change if the velocity changes sign from
% face j-1 to j
for j = 2:nCells-1                          % Does not include boundary effects; maybe change to firstCell to lastCell-1 later
    if (xvel_FC(j) >= 0)
        influxVol  = ofs(j-1);
        outfluxVol = ofs(j);
        q_outflux  = q_slab(j);
        q_influx   = q_slab(j-1);
    end
    if (xvel_FC(j) < 0)
        influxVol  = ofs(j+1);
        outfluxVol = ofs(j);
        q_outflux  = q_slab(j);
        q_influx   = q_slab(j+1);
    end
    q_advected(j) = (q_influx * influxVol - q_outflux*outfluxVol)/cellVol;

    % % Print advected quantities near shock front at first timestep
    %     if ((j >= minPrintRange) & (j <= maxPrintRange))
    %       fprintf('q_influx %E influxVol %E q_outflux %E outfluxVol %E q_advected %E\n',q_influx, influxVol, q_outflux, outfluxVol,q_advected(j));
    %     end
end
