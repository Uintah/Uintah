function y = roundd(x,d)
%ROUNDD  Round towards nearest rational (to certain # of digits).
%   ROUNDD(X,D) rounds the elements of X to the nearest rational A/(10^D *
%   B), where A and B are integers. Alternatively, ROUNDD(X,D) = 10^(-D) *
%   ROUND(10^D * X). ROUNDD(X) is the same as ROUNDD(X,0).
%
%   See also ROUND, FLOOR, CEIL, FIX.

% Revision history:
% 17-JUL-2005    Oren Livne    Created

if (nargin < 2)
    d = 0;
end
f = 10^d;
y = round(x*f)/f;
