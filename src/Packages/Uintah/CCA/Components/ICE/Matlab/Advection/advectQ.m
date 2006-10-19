%____________________________________________________________
% Function:  AdvectQ
% Computes the advection of q
function[q_advected,gradLim,grad_x] = advectQ(q,rho,rho_slab,rho_vrtx_1, rho_vrtx_2, ofs, rx, xvel_FC, delX, nCells)
  
  q_slab =[0:nCells+1];
  [gradLim, grad_x]        = gradientLimiter(q, rho, rho_vrtx_1, rho_vrtx_2, delX, nCells);
  [q_slab,gradLim, grad_x] = qAverageFlux(q, rho, rho_slab, rx, grad_x, gradLim, delX, nCells);
  q_advected = advectSlabs(xvel_FC, q_slab, ofs, nCells);
  clear q_slab;
end

%_____________________________________________________________
% Function gradientLimiter  -- compatible flux Gradient limiter
function[gradLim, grad_x] = gradientLimiter(q,rho, rho_vrtx_1, rho_vrtx_2, delX, nCells)
  fprintf('gradientLimiter\n');
  
  for( j =2:nCells-1)
    
    grad_x(j) = (q(j+1) - q(j-1))/(2.0*delX);
    %-----------q vertex min/max
    d1 = rho(j)/(rho_vrtx_1(j) + 1e-100);
    d2 = rho(j)/(rho_vrtx_2(j) + 1e-100);
    
    %d1 = 1.0; VanLeer limiter
    %d2 = 1.0;
    
    q_vrtx_1 = q(j) + (grad_x(j) * delX/2.0) * d1;
    q_vrtx_2 = q(j) - (grad_x(j) * delX/2.0) * d2;

    q_vrtx_max = max(q_vrtx_1, q_vrtx_2);
    q_vrtx_min = min(q_vrtx_1, q_vrtx_2);

    % ----------q_CC min/max
    q_max = max(q(j+1), q(j-1));
    q_min = min(q(j+1), q(j-1));

    %----------gradient limiter
    frac = (q_max - q(j))/(max( (q_vrtx_max - q(j)), 1e-100) );
    alphaMax = max(0,frac);
    frac = (q(j) - q_min)/(max( (q(j) - q_vrtx_min), 1e-100) );
    alphaMin = max(0,frac);

    tmp        = min(1,alphaMax);
    gradLim(j) = min(tmp, alphaMin);
    
%     if (q_vrtx_max > q_max) | (q_vrtx_min < q_min)
%       fprintf(' j %i q_vrtx_max %e q_max %e q_vrtx_min %e q_min %e \n',j, q_vrtx_max, q_max, q_vrtx_min, q_min);
%       fprintf(' q_vrtx_1 %e d1 %e q_vrtx_2 %e d2 %e \n',q_vrtx_1, d1, q_vrtx_2, d2);
%       fprintf(' rho_vrtx_1 %e rho_vrtx_2 %e rho(j) %e\n',rho_vrtx_1(j), rho_vrtx_2(j), rho(j));
%       fprintf(' q(j-1) %e q(j) %e q(j+1) %e \n', q(j-1), q(j), q(j+1));
%       fprintf(' grad_x %e\n',grad_x(j));
%       fprintf(' gradLim %e \n \n',gradLim(j));
%     end
  end    
end  


%_____________________________________________________________
% Function:  qAverageFlux
% Computes the value of q at each slab volume.
function[q_slab, gradLim, grad_x] = qAverageFlux(q, rho, rho_slab, rx, grad_x, gradLim, delX, nCells)
  clear j;
  fprintf( 'inside qAverageFlux \n');
  for( j =1:nCells-1)
    % compatible flux fomulation
    q_slab(j) = q(j) * rho_slab(j) + rho(j) * grad_x(j) * gradLim(j) * rx(j);
    % van Leer limiter
    %q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);
  end
end
