%_____________________________________________________________
% Function: advectSlabs
% Compute the increment to q in Eq.(3.1.4) of the reference
function q_advected = advectSlabs(xvel_FC, q_slab, ofs, nCells)
clear j;
fprintf('inside advectSlabs\n');
q_advected = zeros(1,nCells);
% Note that the following has to change if the velocity changes sign from face j-1 to j
for j = 2:nCells-1                          % Does not include boundary effects
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
    q_advected(j) = q_influx * influxVol - q_outflux*outfluxVol;
end
