%_____________________________________________________________
%  Function:  OutFluxVol -- computes the slab volume and the 
%  corresponding centroid (rx) on the outflux face
%  This assumes that dy and dz = 1.0;
function[ofs, rx] = OutFluxVol(xvel_FC, delT, dx, nCells)
  clear j;
  fprintf( 'inside OutFluxVol \n');
  for( j =1:nCells)
    if xvel_FC(j) >= 0
      dx_slab = xvel_FC(j) * delT;
      rx(j) = dx(j)/2.0 - dx_slab/2.0;
    end
    
    if xvel_FC(j) < 0
      dx_slab = xvel_FC(j) * delT;
      rx(j) = dx_slab/2.0 - dx(j)/2.0;
    end
    
    %outflux volumes
    dz = 1.0;
    dy = 1.0;
    ofs(j) = dx_slab * dz * dy;
  end
end
