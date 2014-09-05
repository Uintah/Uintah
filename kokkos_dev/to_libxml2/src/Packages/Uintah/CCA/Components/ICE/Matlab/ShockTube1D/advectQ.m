%____________________________________________________________
% Function:  AdvectQ
% Computes the advection of q using the compatible flux limiter.
function [q_advected,gradLim,grad_x] = ...
    advectQ(q,mass,mass_slab,mass_vrtx_1, mass_vrtx_2, ofs, rx, xvel_FC, delX, nCells)
globalParams;

if (P.advectionOrder == 1)
    qOverRho = q;
else
    % For compatible fluxes only:
    % q is written as q=T/mass for this routine - we will
    % limit the gradient of T, not q.
    qOverRho = q ./ mass;
end

[gradLim, grad_x]           = gradientLimiter(qOverRho, mass, mass_vrtx_1, mass_vrtx_2, delX, nCells);
[q_slab,gradLim, grad_x]    = qAverageFlux(qOverRho, mass, mass_slab, rx, grad_x, gradLim, delX, nCells);
q_advected                  = advectSlabs(xvel_FC, q_slab, ofs, nCells, delX);

%_____________________________________________________________
% Function: gradientLimiter
% Compute compatible flux Gradient limiter
function [gradLim, grad_x] = gradientLimiter(q, mass, mass_vrtx_1, mass_vrtx_2, delX, nCells)
globalParams;

if (P.debugSteps)
    fprintf('gradientLimiter()\n');
end
gradLim     = zeros(nCells,1);
grad_x      = zeros(nCells,1);
q_vrtx_1    = zeros(nCells+1,1);
q_vrtx_2    = zeros(nCells+1,1);

for j = 2:nCells-1
    %----------- Trial gradient, central differencing
    grad_x(j) = (q(j+1) - q(j-1))/(2.0*delX);

    %----------- Compute q at vertices before limiting and min/max of these vertex values
    % Compatible flux limiter - nonlinear temp=q/mass distribution
    d1          = mass(j)/(mass_vrtx_1(j) + d_SMALL_NUM);
    d2          = mass(j)/(mass_vrtx_2(j) + d_SMALL_NUM);
    % van Leer limiter, linear q distribution
    %     d1          = 1.0;
    %     d2          = 1.0;

    q_vrtx_1    = q(j) + (grad_x(j) * delX/2.0) * d1;
    q_vrtx_2    = q(j) - (grad_x(j) * delX/2.0) * d2;
    q_vrtx_max  = max(q_vrtx_1, q_vrtx_2);
    q_vrtx_min  = min(q_vrtx_1, q_vrtx_2);

    %---------- Compute q_CC min/max
    q_max       = max(q(j+1), q(j-1));
    q_min       = min(q(j+1), q(j-1));

    %---------- Gradient limiter, pp.13-14
    frac1       = (q_max - q(j) + d_SMALL_NUM)/...
        (q_vrtx_max - q(j) + d_SMALL_NUM);
    alphaMax    = max(0,frac1);                                 % Eq.(3.2.10c)
    frac2       = (q(j) - q_min + d_SMALL_NUM)/...
        (q(j) - q_vrtx_min + d_SMALL_NUM);
    alphaMin    = max(0,frac2);                                 % Eq.(3.2.10d)
    gradLim(j)  = min([1, alphaMax, alphaMin]);                 % Eq.(3.2.10b)

    %gradLim(j) = 1.0;           % Testing - phi=1
end


%_____________________________________________________________
% Function:  qAverageFlux
% Computes the value of q at each slab volume.
% Equation 3.3.3 or the reference
function [q_slab, gradLim, grad_x] = qAverageFlux(q, mass, mass_slab, rx, grad_x, gradLim, delX, nCells)
globalParams;

if (P.debugSteps)
    fprintf('qAverageFlux()\n');
end
q_slab      = zeros(nCells,1);
if (P.advectionOrder == 1)
    for j = 1:nCells                                                % For cells 1 and nCells, disregard contributions in this matlab code (note: grads=0 there)
        q_slab(j) = q(j);                                           % Limiter=0, first order upwind scheme
    end
else
    for j = 1:nCells                                                % For cells 1 and nCells, disregard contributions in this matlab code (note: grads=0 there)
        q_slab(j) = q(j) * mass_slab(j) + mass(j) * grad_x(j) * gradLim(j) * rx(j);   % compatible flux fomulation; here q = energy (NOT temperature)
        %    q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);                          % van Leer limiter
    end
end

%===================
% PRINTOUTS
if (P.debugAdvectQ)
    fprintf('End of AdvectQ():\n');
    range = P.printRange;   
    
    fprintf('rx = ');
    fprintf('%.10f ' ,rx(range));
    fprintf('\n');

    fprintf('gradLim = ');
    fprintf('%.10f ',gradLim(range));
    fprintf('\n');

    fprintf('grad_x =');
    fprintf('%10f ',grad_x(range));
    fprintf('\n');

    fprintf('q = ');
    fprintf('%.10f ',q(range));
    fprintf('\n');

    fprintf('mass_slab = ');
    fprintf('%.10f ',mass_slab(range));
    fprintf('\n');

    fprintf('mass = ');
    fprintf('%.10f ',mass(range));
    fprintf('\n');

    fprintf('q_slab = ');
    fprintf('%.10f ',q_slab(range));
    fprintf('\n');
end
