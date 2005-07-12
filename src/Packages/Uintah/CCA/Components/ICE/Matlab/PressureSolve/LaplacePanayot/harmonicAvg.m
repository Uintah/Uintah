function a = harmonicAvg(y,z)
% Integral of 1/a along the curve [x1,x2]. Here x1,x2 are x-coordinates if
% dim=1, and y is the fixed y-coordinate along the curve. If dim=2, x1,x2
% are y-coordinates and y is the fixed x-coordinate along the curve.

%HARMONICAVG  Harmonic average of the diffusion coefficient.
%   A = HARMONICAVG(Y,Z) returns the harmonic average of the diffusion
%   coefficient a(X) (X in R^D) along the line connecting the points Y,Z in
%   R^D. That is,
%   A = 1/(integral_0^1 1/a(x1(s),...,xD(s)) ds),
%   where xd(s) = y(d) + s*(z(d) - y(d)) is the arclength parameterization
%   of the d-coordinate of the line y-z, d = 1...D. A is computed
%   analytically for the specific cases we consider; in general, use some
%   simple quadrature formula for A from discrete a-values. See "help
%   exactSolution" for a list of possible problems, determined by
%   param.problemType.
%
%   See also: DIFFUSION, SETUPPATCHINTERIOR, SETUPPATCHINTERFACE.

% Revision history:
% 12-JUL-2005    Oren Livne    Created

r = 0.0;

switch (param.problemType)
    % Problems with a=1
    case {'quadratic','sinsin','GaussianSource','Lshaped'},
        r = 1.0;

    case 'jump',
        % Piecewise constant diffusion coefficient with a big jump at x=0.5.
        switch (dim)
            case 1,
                bothLeft        = find((x1 < 0.5) & (x2 < 0.5));
                r(bothLeft)     = x2(bothLeft)-x1(bothLeft);

                crossJump       = find((x1 < 0.5) & (x2 > 0.5));
                r(crossJump)    = (0.5-x1(crossJump)) + (x2(crossJump)-0.5)/1e+6;
                
                bothRight       = find(x1 > 0.5);
                r(bothRight)    = (x2(bothRight)-x1(bothRight))/1e+6;
            case 2,
                r = x2-x1;
        end

    otherwise,
        error('Unknown problem type');
end
