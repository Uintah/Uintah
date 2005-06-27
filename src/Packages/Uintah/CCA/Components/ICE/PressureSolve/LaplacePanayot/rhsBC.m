function g = rhsBC(x,y)
% For Dirichlet B.C. : u = g at domain boundaries
[xmat1,xmat2] = ndgrid(x,y);

g = zeros(size(xmat1));
%g = xmat1;
