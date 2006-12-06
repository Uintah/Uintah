%____________________________________________________________
% Function:  AdvectQ
% Computes the advection of q
function[q_advected,gradLim,grad_x] = advectQ(q,mass,mass_slab,mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, dx, nCells,advOrder)
  
  %convert to primitive variables
  for( j =1:nCells)
    q(j) = q(j)/mass(j);
  end
  
  q_slab =[0:nCells+1];
  [gradLim, grad_x]        = gradientLimiter(q, mass, mass_vrtx_1, mass_vrtx_2, dx, nCells);
  [q_slab,gradLim, grad_x] = qAverageFlux(q, mass, mass_slab, rx, grad_x, gradLim, dx, nCells, advOrder);
  q_advected = advectSlabs(xvel_FC, q_slab, ofs, nCells,dx);
  clear q_slab;
end

%_____________________________________________________________
% Function gradientLimiter  -- compatible flux Gradient limiter
function[gradLim, grad_x] = gradientLimiter(q,mass, mass_vrtx_1, mass_vrtx_2, dx, nCells)
  fprintf('gradientLimiter\n');
  
  for( j =2:nCells)
      
    %central difference non-uniform spacing  
    alpha = dx(j+1)/dx(j-1);
    numerator   = q(j+1) + (alpha^2 - 1.0)* q(j) - alpha^2 * q(j-1);
    denominator = alpha * (alpha + 1.0) * dx(j-1);
    grad_x(j) = numerator/denominator;
      
    % central difference uniform spacing
    %grad_x(j) = (q(j+1) - q(j-1))/(2.0*dx(j));
    
    % Test of backward differencing at the CFI
    if(dx(j+1) > dx(j))  % Right CFI 
      grad_x(j) = ( q(j) - q(j-1) )/dx(j);
    end
    if(dx(j+1) < dx(j))  % Left CFI 
      grad_x(j) = ( q(j+1) - q(j) )/dx(j);
    end
    
    smallNum = 1e-100;
    %-----------q vertex min/max
    d1 = mass(j)/(mass_vrtx_1(j) + smallNum);
    d2 = mass(j)/(mass_vrtx_2(j) + smallNum);
    
    q_vrtx_1 = q(j) + (grad_x(j) * dx(j)/2.0) * d1;
    q_vrtx_2 = q(j) - (grad_x(j) * dx(j)/2.0) * d2;

    q_vrtx_max = max(q_vrtx_1, q_vrtx_2);
    q_vrtx_min = min(q_vrtx_1, q_vrtx_2);

    % ----------q_CC min/max
    q_max = max(q(j+1), q(j-1));
    q_min = min(q(j+1), q(j-1));

    %----------gradient limiter
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
    
    %----------Test of clamping the limiter at the CFI
    if( (dx(j+1)/dx(j)) ~= 1)
      fprintf(' j %i alpha: %e turned off the gradient limiter\n',j, alpha);
      gradLim(j) = 0.0;
    end 
        
%     if (q_vrtx_max > q_max) | (q_vrtx_min < q_min)
%       fprintf(' j %i q_vrtx_max %e q_max %e q_vrtx_min %e q_min %e \n',j, q_vrtx_max, q_max, q_vrtx_min, q_min);
%       fprintf(' q_vrtx_1 %e d1 %e q_vrtx_2 %e d2 %e \n',q_vrtx_1, d1, q_vrtx_2, d2);
%       fprintf(' mass_vrtx_1 %e mass_vrtx_2 %e mass(j) %e\n',mass_vrtx_1(j), mass_vrtx_2(j), mass(j));
%       fprintf(' q(j-1) %e q(j) %e q(j+1) %e \n', q(j-1), q(j), q(j+1));
%       fprintf(' grad_x %e\n',grad_x(j));
%       fprintf(' gradLim %e \n \n',gradLim(j));
%     end
  end    
end  


%_____________________________________________________________
% Function:  qAverageFlux
% Computes the value of q at each slab volume.
function[q_slab, gradLim, grad_x] = qAverageFlux(q, mass, mass_slab, rx, grad_x, gradLim, dx, nCells,advOrder)
  clear j;
  fprintf( 'inside qAverageFlux: Order %i\n',advOrder);
  for( j =1:nCells)
    % compatible flux fomulation
    if(advOrder == 2)   % second order
      q_slab(j) = q(j) * mass_slab(j) + mass(j) * grad_x(j) * gradLim(j) * rx(j);
    end
    if(advOrder == 1)   % first order
      q_slab(j) = q(j) * mass_slab(j);
    end
  end
end
