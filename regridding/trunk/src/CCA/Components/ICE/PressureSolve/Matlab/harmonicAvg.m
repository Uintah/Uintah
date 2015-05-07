function a = harmonicAvg(x,y,z)
%HARMONICAVG  Harmonic average of the diffusion coefficient.
%   A = HARMONICAVG(X,Y,Z) returns the harmonic average of the diffusion
%   coefficient a(T) (T in R^D) along the line connecting the points X,Y in
%   R^D. That is,
%   A = 1/(integral_0^1 1/a(t1(s),...,tD(s)) ds),
%   where td(s) = x{d} + s*(y{d} - x{d})/norm(y-x) is the arclength
%   parameterization of the d-coordinate of the line x-y, d = 1...D.  We
%   assume that A is piecewise constant with jump at Z (X,Y are normally
%   cell centers and Z at the cell face). X,Y,Z are Dx1 cell arrays,
%   we treat every element of X{d},Y{d},Z{d} separately and output A as an
%   array of size size(X{1}).
%   In general, A can be analytically computedfor the specific cases
%   we consider; in general, use some simple quadrature formula for A 
%   from discrete a-values.
%   See "help exactSolution" for a list of possible problems, determined by
%   param.problemType.
%
%   ### NOTE: ### If we use a different refinement ratio in different
%   dimensions, near the interface we may need to compute A along lines X-Y
%   that cross more than one cell boundary. This is currently ignored and
%   we assume all lines cut one cell interface only.
%
%   See also: DIFFUSION, SETUPPATCHINTERIOR, SETUPPATCHINTERFACE.

% Revision history:
% 12-JUL-2005    Oren Livne    Created
% 13-JUL-2005    Oren Livne    Piecewise constant model for a.

globalParams;

d                   = length(x);
a                   = zeros(size(x{1}));

Ax                  = diffusion(x);
Ay                  = diffusion(y);

dxy                 = zeros(size(x{1}));
dxz                 = zeros(size(x{1}));
for dim = 1:d,
    dxy             = dxy + abs(y{dim}-x{dim}).^2;
    dxz             = dxz + abs(z{dim}-x{dim}).^2;   
end
K                   = sqrt(dxz./dxy);
a                   = (Ax.*Ay)./((1-K).*Ax + K.*Ay);
