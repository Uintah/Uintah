%____________________________________________________________
% Function:  AdvectQ
% Computes the advection of q using the compatible flux limiter.
function [q_advected,gradLim,grad_x] = ...
    advectQ(q,rho,rho_slab,rho_vrtx_1, rho_vrtx_2, ofs, rx, xvel_FC, delX, nCells)
globalParams;

% If compatible fluxes: q will be T=Energy/rho for this routine, as we want to limit
% the gradient of T
qOverRho = q ./ (rho + d_SMALL_NUM);

[gradLim, grad_x]           = gradientLimiter(qOverRho, rho, rho_vrtx_1, rho_vrtx_2, delX, nCells);
[q_slab,gradLim, grad_x]    = qAverageFlux(qOverRho, rho, rho_slab, rx, grad_x, gradLim, delX, nCells);
q_advected                  = advectSlabs(xvel_FC, q_slab, ofs, nCells);

%_____________________________________________________________
% Function: gradientLimiter
% Compute compatible flux Gradient limiter
function [gradLim, grad_x] = gradientLimiter(q, rho, rho_vrtx_1, rho_vrtx_2, delX, nCells)
globalParams;

fprintf('gradientLimiter\n');
gradLim     = zeros(nCells,1);
grad_x      = zeros(nCells,1);
q_vrtx_1    = zeros(nCells+1,1);
q_vrtx_2    = zeros(nCells+1,1);

for j = 2:nCells-1    
    %----------- Trial gradient, central differencing
    grad_x(j) = (q(j+1) - q(j-1))/(2.0*delX);
    
    %----------- Compute q at vertices before limiting and min/max of these vertex values
    %     % Compatible flux limiter - nonlinear temp=q/rho distribution
    %     d1          = rho(j)/(rho_vrtx_1(j) + 1e-100);
    %     d2          = rho(j)/(rho_vrtx_2(j) + 1e-100);
    % van Leer limiter, linear q distribution
    d1          = 1.0;
    d2          = 1.0;

    q_vrtx_1    = q(j) + (grad_x(j) * delX/2.0) * d1;
    q_vrtx_2    = q(j) - (grad_x(j) * delX/2.0) * d2;
    q_vrtx_max  = max(q_vrtx_1, q_vrtx_2);
    q_vrtx_min  = min(q_vrtx_1, q_vrtx_2);
    
    %---------- Compute q_CC min/max
    q_max       = max(q(j+1), q(j-1));
    q_min       = min(q(j+1), q(j-1));
    
    %---------- Gradient limiter, pp.13-14
    frac1       = (q_max - q(j))/(max( (q_vrtx_max - q(j)), 1e-100) );
    alphaMax    = max(0,frac1);                                 % Eq.(3.2.10c)
    frac2       = (q(j) - q_min)/(max( (q(j) - q_vrtx_min), 1e-100) );
    alphaMin    = max(0,frac2);                                 % Eq.(3.2.10d)    
    gradLim(j)  = min([1, alphaMax, alphaMin]);                 % Eq.(3.2.10b)
end    


%_____________________________________________________________
% Function:  qAverageFlux
% Computes the value of q at each slab volume.
% Equation 3.3.3 or the reference
function [q_slab, gradLim, grad_x] = qAverageFlux(q, rho, rho_slab, rx, grad_x, gradLim, delX, nCells)

fprintf('Inside qAverageFlux\n');
q_slab      = zeros(nCells,1);
for j = 1:nCells                                                % For cells 1 and nCells, disregard contributions in this matlab code (note: grads=0 there)
    q_slab(j) = q(j);                                           % Limiter=0, first order upwind scheme
    %    q_slab(j) = q(j) * rho_slab(j) + rho(j) * grad_x(j) * gradLim(j) * rx(j);   % compatible flux fomulation; here q = energy (NOT temperature)
%    q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);                          % van Leer limiter
end
