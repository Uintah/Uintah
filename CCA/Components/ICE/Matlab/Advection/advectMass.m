%_____________________________________________________________
% Function:   advectMass
% computes the advection of mass using vanLeer Limiter
function[q_advected,gradLim,grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = advectMass( mass,ofs, rx,xvel_FC, dx, nCells)
  
  [gradLim, grad_x, mass_vrtx_1, mass_vrtx_2] = gradientLimiter_Mass(mass, dx, nCells);
  [mass_slab, gradLim, grad_x]                = qAverageFluxMass(mass, rx, grad_x, gradLim, nCells);
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

    %-----------q vertex min/max
    q_vrtx_1_tmp = q(j) + grad_x(j) * dx(j)/2.0;
    q_vrtx_2_tmp = q(j) - grad_x(j) * dx(j)/2.0;

    q_vrtx_max = max(q_vrtx_1_tmp, q_vrtx_2_tmp);
    q_vrtx_min = min(q_vrtx_1_tmp, q_vrtx_2_tmp);

    % ----------q_CC min/max
    q_max = max(q(j+1), q(j-1));
    q_min = min(q(j+1), q(j-1));

    %----------gradient limiter
    smallNum = 1e-100;
    frac1    = (q_max - q(j) + smallNum)/(q_vrtx_max - q(j) + smallNum );
    alphaMax = max(0,frac1);
    frac2    = (q(j) - q_min + smallNum)/(q(j) - q_vrtx_min + smallNum );
    alphaMin = max(0,frac2);

    tmp        = min(1,alphaMax);
    gradLim(j) = min(tmp, alphaMin);
       
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
function[q_slab, gradLim, grad_x] = qAverageFluxMass(q, rx, grad_x, gradLim, nCells)
  clear j;
  fprintf( 'inside qAverageFluxRho \n');
  for( j =1:nCells)
    q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);
%   fprintf(' j %i  q_slab: %e q(j): %e grad_x(j): %e  gradLim(j): %e  rx(j): %e\n',j, q_slab(j),q(j), grad_x(j), gradLim(j), rx(j));
  end
end
