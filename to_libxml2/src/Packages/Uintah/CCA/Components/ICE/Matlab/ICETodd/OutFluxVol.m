%_____________________________________________________________
%  Function:  OutFluxVol -- computes the slab volume and the 
%  corresponding centroid (rx) on the outflux face
%  This assumes that dy and dz = 1.0;
function[ofs, rx] = OutFluxVol(xvel_FC, delT, delX, nCells)
  clear j;
  fprintf( 'inside OutFluxVol \n');
  for( j =1:nCells)
    if xvel_FC(j) >= 0
      delX_slab = xvel_FC(j) * delT;
      rx(j) = delX/2.0 - delX_slab/2.0;
    end
    
    if xvel_FC(j) < 0
      delX_slab = xvel_FC(j) * delT;
      rx(j) = delX_slab/2.0 - delX/2.0;
    end
    
    %outflux volumes
    delY_Z = 1.0;
    ofs(j) = delX_slab * delY_Z;
  end
end
