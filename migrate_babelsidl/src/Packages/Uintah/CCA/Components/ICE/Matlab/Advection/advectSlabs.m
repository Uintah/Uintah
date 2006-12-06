%_____________________________________________________________
function[q_advected] = advectSlabs(xvel_FC, q_slab, ofs, nCells,dx)
  clear j;
  fprintf( 'inside advectSlabs');
  for( j =2:nCells-1)
    % flow is to the right
    if (xvel_FC(j) >= 0)
      influxVol  = ofs(j-1);
      outfluxVol = ofs(j);
      q_outflux  = q_slab(j);
      q_influx   = q_slab(j-1);
    end
    % flow is to left
    if (xvel_FC(j) < 0)
      influxVol  = ofs(j+1);
      outfluxVol = ofs(j);
      q_outflux  = q_slab(j);
      q_influx   = q_slab(j+1);
    end

    q_advected(j) = (q_influx * influxVol - q_outflux*outfluxVol)/dx(j);
        
    %fprintf(' j %i  q_advected: %e q_influx: %e influxVol: %e q_outflux: %e  outfluxVol: %e \n',j, q_advected(j),q_influx,influxVol,q_outflux,outfluxVol);
  end
end
