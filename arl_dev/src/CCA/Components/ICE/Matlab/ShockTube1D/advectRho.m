%_____________________________________________________________
% Function: advectRho
% computes the advection of mass using vanLeer Limiter
function [q_advected,gradLim,grad_x, mass_slab, mass_vrtx_1, mass_vrtx_2] = ...
  advectRho( mass,ofs, rx,xvel_FC, G)
globalParams;

[gradLim, grad_x, mass_vrtx_1, mass_vrtx_2]  = gradientLimiterRho(mass, G);
[mass_slab, gradLim, grad_x]                 = qAverageFluxRho(mass, rx, grad_x, gradLim,G);
q_advected                                   = advectSlabs(xvel_FC, mass_slab, ofs,G);

%_____________________________________________________________
% Function: gradientLimiterRho
% Computes standard the gradient, the vanLeer limiter
% and vertex values of mass (q_vertex1, q_vrtx2).  The vertex values
% are needed to compute the compatible flux limiter.
function [gradLim, grad_x, q_vrtx_1, q_vrtx_2] = gradientLimiterRho(q,G)
globalParams;

if (P.debugSteps)
  fprintf('gradientLimiterRho()\n');
end
gradLim     = zeros(G.ghost_Left, G.ghost_Right);
grad_x      = zeros(G.ghost_Left, G.ghost_Right);
q_vrtx_1    = zeros(G.ghost_Left, G.ghost_Right);
q_vrtx_2    = zeros(G.ghost_Left, G.ghost_Right);
dx          = G.delX;

for j = G.first_CC:G.last_CC
  %----------- Trial gradient, central differencing
  grad_x(j) = (q(j+1) - q(j-1))./(2.0*dx);

  %----------- Compute q at vertices before limiting and min/max of these vertex values
  q_vrtx_1_tmp = q(j) - grad_x(j) * dx/2.0;                 % Eq.(3.2.8) for left vertex value
  q_vrtx_2_tmp = q(j) + grad_x(j) * dx/2.0;                 % Eq.(3.2.8) for right vertex value
  q_vrtx_max  = max(q_vrtx_1_tmp, q_vrtx_2_tmp);
  q_vrtx_min  = min(q_vrtx_1_tmp, q_vrtx_2_tmp);

  %---------- Compute q_CC min/max
  q_max = max(q(j+1), q(j-1));
  q_min = min(q(j+1), q(j-1));

  %---------- Gradient limiter, pp.13-14
  frac1       = (q_max - q(j) + d_SMALL_NUM)/(q_vrtx_max - q(j) + d_SMALL_NUM);
  alphaMax    = max(0.0,frac1);                               % Eq.(3.2.10c)
   
  frac2       = (q_min - q(j) + d_SMALL_NUM)/(q_vrtx_min  - q(j)  + d_SMALL_NUM);
  alphaMin    = max(0.0,frac2);                               % Eq.(3.2.10d)
                                           
  gradLim(j)  = min([1, alphaMax, alphaMin]);              

  %---------- Save vertex values after gradient limiting, for advectQ
  q_vrtx_1(j) = q(j) - (grad_x(j) * gradLim(j) * dx/2.0);   % Eq.(3.2.8) for left vertex value
  q_vrtx_2(j) = q(j) + (grad_x(j) * gradLim(j) * dx/2.0);   % Eq.(3.2.8) for right vertex value

  %gradLim(j) = 1.0;           % Testing - phi=1
end

%_____________________________________________________________
% Function: qAverageFluxRho
% computes Q in the slab according to the reference.
function [q_slab, gradLim, grad_x] = qAverageFluxRho(q, rx, grad_x, gradLim,G)
globalParams;

q_slab = zeros(G.ghost_Left,G.ghost_Right);
if (P.advectionOrder == 1)
  for j = G.ghost_Left:G.ghost_Right                            % For cells 1 and nCells, disregard contributions in this matlab code (note: grads=0 there)
    q_slab(j) = q(j);                                           % Limiter=0, first order upwind scheme
  end
else
  for j = G.ghost_Left:G.ghost_Right                            % For cells 1 and nCells, disregard contributions in this matlab code (note: grads=0 there)
    q_slab(j) = q(j) + grad_x(j) * gradLim(j) * rx(j);          % Eq.(3.2.9)
  end
end

%===================
% PRINTOUTS
if (P.debugAdvectRho)

  for j = G.ghost_Left:G.ghost_Right
    joffset = j-2;
    if(joffset >26 && joffset < 33)
      fprintf('%i RX: %16.15E gradX %16.15E q(j) %16.15E q_slab %16.15E\n', joffset, rx(j), grad_x(j)*gradLim(j),q(j), q_slab(j));
    end
  end
end
