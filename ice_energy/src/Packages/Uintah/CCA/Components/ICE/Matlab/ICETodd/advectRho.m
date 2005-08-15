%_____________________________________________________________
% Function:   advectRho
% computes the advection of rho using vanLeer Limiter
function[q_advected,gradLim,grad_x, rho_slab, rho_vrtx_1, rho_vrtx_2] = advectRho( rho,ofs, rx,xvel_FC, delX, nCells)
  
  [gradLim, grad_x, rho_vrtx_1, rho_vrtx_2] = gradientLimiter_Rho(rho, delX, nCells);
  [rho_slab, gradLim, grad_x]  = qAverageFluxRho(rho, rx, grad_x, gradLim, nCells);
  q_advected = advectSlabs(xvel_FC, rho_slab, ofs, nCells);

end

%_____________________________________________________________
% Function gradientLimiter_Rho
% Computes standard the gradient, the vanLeer limiter 
% and vertex values of rho (q_vertex1, q_vrtx2).  The vertex values
% are needed to compute the compatible flux limiter.
function[gradLim, grad_x, q_vrtx_1, q_vrtx_2] = gradientLimiter_Rho(q,delX, nCells)
  fprintf('gradientLimiter_ Rho\n');
  for( j =2:nCells-1)
    grad_x(j) = (q(j+1) - q(j-1))./(2.0*delX);

    %-----------q vertex min/max
    q_vrtx_1_tmp = q(j) + grad_x(j) * delX/2.0;
    q_vrtx_2_tmp = q(j) - grad_x(j) * delX/2.0;

    q_vrtx_max = max(q_vrtx_1_tmp, q_vrtx_2_tmp);
    q_vrtx_min = min(q_vrtx_1_tmp, q_vrtx_2_tmp);

    % ----------q_CC min/max
    q_max = max(q(j+1), q(j-1));
    q_min = min(q(j+1), q(j-1));

    %----------gradient limiter
    frac1 = (q_max - q(j))/(max( (q_vrtx_max - q(j)), 1e-100) );
    alphaMax = max(0,frac1);
    frac2 = (q(j) - q_min)/(max( (q(j) - q_vrtx_min), 1e-100) );
    alphaMin = max(0,frac2);

    tmp        = min(1,alphaMax);
    gradLim(j) = min(tmp, alphaMin);
        
    q_vrtx_1(j) = q(j) + (grad_x(j) * gradLim(j) * delX/2.0);
    q_vrtx_2(j) = q(j) - (grad_x(j) * gradLim(j) * delX/2.0);
      
%     if (q_vrtx_max > q_max) | (q_vrtx_min < q_min)
%       fprintf(' j %i grad_x %e gradLim %e alphaMax %e alphaMin %e\n',j, grad_x(j), gradLim(j), alphaMax, alphaMin);
%       fprintf(' q %e q_max %e q_min %e, q_vrtx_max %e q_vrtx_min %e\n', q(j), q_max, q_min, q_vrtx_max, q_vrtx_min);
%       fprintf(' frac1 %e frac2 %e \n',frac1, frac2);
%       fprintf(' q_vrtx_1 %e q_vrtx_2 %e \n\n',q_vrtx_1(j), q_vrtx_2(j));
%     end
  end
end
  
  
%_____________________________________________________________
% Function:  qAverageFluxRho
% computes Q in the slab according to the reference. 
function[q_slab, gradLim, grad_x] = qAverageFluxRho(q, rx, grad_x, gradLim, nCells)
  clear j;
  fprintf( 'inside qAverageFluxRho \n');
  for( j =1:nCells-1)
    q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);
  end
end
