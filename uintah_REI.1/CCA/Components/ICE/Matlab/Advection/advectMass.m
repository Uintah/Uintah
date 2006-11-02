%_____________________________________________________________
% Function:   advectMass
% computes the advection of mass using vanLeer Limiter
function[q_advected,gradLim,grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = advectMass( mass,ofs, rx,xvel_FC, dx, nCells, advOrder)
  
  [gradLim, grad_x, mass_vrtx_1, mass_vrtx_2] = gradientLimiter_Mass(mass, dx, nCells);
  [mass_slab, gradLim, grad_x]                = qAverageFluxMass(mass, rx, grad_x, gradLim, nCells,advOrder);
  q_advected                                  = advectSlabs(xvel_FC, mass_slab, ofs, nCells,dx);

end

%_____________________________________________________________
% Function gradientLimiter_Mass
% Computes standard the gradient, the vanLeer limiter 
% and vertex values of mass (q_vertex1, q_vrtx2).  The vertex values
% are needed to compute the compatible flux limiter.
function[gradLim, grad_x, q_vrtx_1, q_vrtx_2] = gradientLimiter_Mass(q,dx, nCells)
  fprintf('gradientLimiter_Mass\n');
  for( j =2:nCells)
  
    %central difference non-uniform spacing
    alpha       = dx(j+1)/dx(j-1);
    numerator   = q(j+1) + (alpha^2 - 1.0)* q(j) - alpha^2 * q(j-1);
    denominator = alpha * (alpha + 1.0) * dx(j-1);
    grad_x(j)   = numerator/denominator;
  
    %central difference uniform spacing 
    %grad_x(j) = (q(j+1) - q(j-1))/(2.0*dx(j));
    
    
    % Test of backward differencing at the CFI
    if(dx(j+1) > dx(j))  % Right CFI 
      grad_x(j) = ( q(j) - q(j-1) )/dx(j); 
    end
    if(dx(j+1) < dx(j))  % Left CFI 
      grad_x(j) = ( q(j+1) - q(j) )/dx(j);
    end

    %-----------q vertex min/max
    q_vrtx_1_tmp = q(j) + grad_x(j) * dx(j)/2.0;
    q_vrtx_2_tmp = q(j) - grad_x(j) * dx(j)/2.0;

    q_vrtx_max = max(q_vrtx_1_tmp, q_vrtx_2_tmp);
    q_vrtx_min = min(q_vrtx_1_tmp, q_vrtx_2_tmp);

    % ----------q_CC min/max
    q_max = max(q(j+1), q(j-1));
    q_min = min(q(j+1), q(j-1));

    %----------gradient limiter
%     smallNum = 1e-100;
%     frac1    = (q_max - q(j) + smallNum)/(q_vrtx_max - q(j) + smallNum );
%     alphaMax = max(0,frac1);
%     frac2    = (q(j) - q_min + smallNum)/(q(j) - q_vrtx_min + smallNum );
%     alphaMin = max(0,frac2);

    %----------CFDLib gradient limiter
    frac = (q_max - q(j))/(max( (q_vrtx_max - q(j)), 1e-100) );
    alphaMax = max(0,frac);
    frac = (q(j) - q_min)/(max( (q(j) - q_vrtx_min), 1e-100) );
    alphaMin = max(0,frac);

    tmp        = min(1,alphaMax);
    gradLim(j) = min(tmp, alphaMin);
    
    %----------test of clamping the limiter at the CFI  
    if( (dx(j+1)/dx(j)) ~= 1)
      fprintf(' j %i alpha: %e turned off the gradient limiter\n',j, alpha);
      gradLim(j) = 0.0;
    end  
       
    %-----------mass vertex (limited)
    q_vrtx_1(j) = q(j) + (grad_x(j) * gradLim(j) * dx(j)/2.0);
    q_vrtx_2(j) = q(j) - (grad_x(j) * gradLim(j) * dx(j)/2.0);
      
%     if (q_vrtx_max > q_max) | (q_vrtx_min < q_min) 
%       fprintf(' j %i grad_x %e gradLim %e alphaMax %e alphaMin %e\n',j, grad_x(j), gradLim(j), alphaMax, alphaMin);
%       fprintf(' q %e q_max %e q_min %e, q_vrtx_max %e q_vrtx_min %e\n', q(j), q_max, q_min, q_vrtx_max, q_vrtx_min);
%       fprintf(' q(j-1) %e q(j) %e q(j+1) %e \n', q(j-1), q(j), q(j+1));
%       fprintf(' frac1 %e frac2 %e \n',frac1, frac2);
%       fprintf(' q_vrtx_1 %e q_vrtx_2 %e \n\n',q_vrtx_1(j), q_vrtx_2(j));
%     end
    
  end
end
  
  
%_____________________________________________________________
% Function:  qAverageFluxMass
% computes Q in the slab according to the reference. 
function[q_slab, gradLim, grad_x] = qAverageFluxMass(q, rx, grad_x, gradLim, nCells, advOrder)
  clear j;
  fprintf( 'inside qAverageFluxMass: Order %i \n',advOrder);
  
  
  for( j =1:nCells)
    if(advOrder == 2)  %second order
      q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);
    end
    if(advOrder == 1)  %first order
      q_slab(j) = q(j);
    end
  end
end
