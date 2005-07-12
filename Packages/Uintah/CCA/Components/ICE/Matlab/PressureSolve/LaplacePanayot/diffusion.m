function a = diffusion(x,y)
%DIFFUSION  Diffusion coefficient.
%   A = DIFFUSION(TYPE,X,Y) returns the value of the diffusion coefficient
%   at (X,Y). X and Y can be matrices and then A is a corresponding
%   matrix of values (works for any dimensional array in fact).
%   PARAM.problemType selects the problem type. See "help exactsolution"
%   for a list of possible problems.
%
%   See also: EXACTSOLUTION, RHS, RHSBC.

% Revision history:
% 12-JUL-2005    Oren Livne    Created

globalParams;

a = zeros(size(x));

switch (param.problemType)
    case 'smooth',
        % Constant diffusion
        a = repmat(1,size(x));

    case 'jump',
        % Piecewise constant diffusion coefficient with a big jump at x=0.5.
        left = find(x < 0.5);
        right = find(x > 0.5);
        a(left) = 1.0;
        a(right) = 1.0e+6;
end
