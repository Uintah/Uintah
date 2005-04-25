%_____________________________________________________________
function[q_advected] = advectSlabs(xvel_FC, q_slab, ofs, nCells)
  clear j;
  fprintf( 'inside advectSlabs');
  for( j =2:nCells-1)
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
end
